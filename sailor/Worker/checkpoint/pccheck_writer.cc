// pccheck simplified version - 1 concurrent checkpoint, no pipelining

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <atomic>
#include <cstring>
#include <unistd.h>
#include <fstream>
#include <pthread.h>
#include <sys/time.h>
#include <chrono>
#include <assert.h>
#include <stdio.h>
#include <linux/mman.h>
#include <sys/types.h>
#include <fcntl.h> /* Definition of AT_* constants */
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <atomic>

using namespace std;
#define CACHELINES_IN_1G 16777216
#define BYTES_IN_1G 1073741824
#define CACHELINE_SIZE 64
#define OFFSET_SIZE 4096
#define FLOAT_IN_CACHE 16
#define REGION_SIZE 113406487152ULL
#define MAX_ITERATIONS 8

static int *PR_ADDR;

static uint8_t Cores[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

struct thread_data
{
    uint32_t id;
    float *arr;
    float *pr_arr;
    uint32_t size;
} __attribute__((aligned(64)));

static int SIZE = 512;

/* Allocate one core per thread */
static inline void set_cpu(int cpu)
{
    assert(cpu > -1);
    int n_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu < n_cpus)
    {
        int cpu_use = Cores[cpu];
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu_use, &mask);
        pthread_t thread = pthread_self();
        if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &mask) != 0)
        {
            fprintf(stderr, "Error setting thread affinity\n");
        }
    }
}


static void mapFile(const char *filename, int *regionAddr, const uint64_t regionSize, bool data, int fd)
{
    if ((PR_ADDR = (int *)mmap(NULL, regionSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED)
    {
        perror("mmap_file");
        exit(1);
    }
    printf("File mapped at %p\n", PR_ADDR);
}

static void FLUSH(void *p)
{
    asm volatile("clwb (%0)" ::"r"(p));
}

static void SFENCE()
{
    asm volatile("sfence" ::: "memory");
}

static void BARRIER(void *p)
{
    FLUSH(p);
    SFENCE();
}

static void initialize(const char *filename)
{

    struct stat buffer;
    bool newfile = (stat(filename, &buffer) == -1);

    int fd = open(filename, O_CREAT | O_RDWR | O_TRUNC, (mode_t)0666);
    if (fd < 0) {
        perror("Invalid file descriptor");
        exit(1);
    }
    ftruncate(fd, REGION_SIZE);

    //mapPersistentRegion(filename, PR_ADDR_DATA, REGION_SIZE, true, fd);
    mapFile(filename, PR_ADDR, REGION_SIZE, false, fd);

}

class NVM_write
{
public:
    static void savenvm_thread_nd(thread_data *data)
    {
        int id = data->id;
        float *arr = data->arr;
        float *add = (float *)data->pr_arr;
        size_t sz = data->size;

        set_cpu(id);
        printf("At savenvm_thread_nd id is %d, sz is %lu, start address is %p!\n", id, sz, add);
        for (size_t i = 0; i < sz;)
        {

            memcpy((void *)add, (void *)arr, SIZE);
            arr += SIZE / sizeof(float);
            add += SIZE / sizeof(float);
            i += SIZE / sizeof(float);
        }
    }

    static void savenvmNew(float *arr, size_t total_size, int num_threads, int parall_iter)
    {

        float *curr_arr = arr; // address of the current batch
        size_t size_for_thread = total_size / num_threads;
        size_t reminder = total_size % num_threads;

        // make sure the start address is aligned at 4KB
        size_t offset = OFFSET_SIZE - (total_size * sizeof(float)) % OFFSET_SIZE;
        // printf("offset is %ld\n", offset);
        float *curr_pr_arr = (float *)PR_ADDR + (parall_iter * total_size) + (parall_iter * offset) / 4;

        thread *threads[num_threads];
        thread_data allThreadsData[num_threads];

        size_t num_floats_SIZE = SIZE / sizeof(float);
        size_t rem_floats_SIZE = size_for_thread % num_floats_SIZE;
        size_t curr_sz = 0;

        printf("Save checkpoint - create threads!\n");

        for (int i = 0; i < num_threads; i++)
        {
            size_t size_for_thread_i = size_for_thread;
            // all should be multiple of SIZE
            size_for_thread_i += num_floats_SIZE - rem_floats_SIZE;
            size_for_thread_i = std::min(size_for_thread_i, total_size - curr_sz);

            thread_data &data = allThreadsData[i];

            // take into a consideration all the running threads in the system
            data.id = parall_iter * num_threads + i + 1;
            // the address to copy from
            data.arr = curr_arr;
            // the address to copy to
            data.pr_arr = curr_pr_arr;
            data.size = size_for_thread_i;
            threads[i] = new thread(&savenvm_thread_nd, &data);
            curr_arr += size_for_thread_i;
            curr_pr_arr += size_for_thread_i;
            curr_sz += size_for_thread_i;
        }

        for (int j = 0; j < num_threads; j++)
        {
            threads[j]->join();
        }

        // do a total msync here
        printf("curr_pr_arr is %p\n", curr_pr_arr);
        auto t1 = std::chrono::high_resolution_clock::now();
        int res = msync((void *)(curr_pr_arr), total_size * sizeof(float), MS_SYNC);
        if (res == -1)
        {
            printf("[ERROR] Proc %d msync during model persisting addr %p, size %lu\n", parall_iter, curr_pr_arr, total_size * sizeof(float));
            exit(1);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        printf("MSYNC TOOK %f ms\n", ms_double.count());

        return;
    }
};

//====================================================================

extern "C"
{

    NVM_write *writer(const char *filename)
    {
        NVM_write *nvmobj = new NVM_write();
        initialize(filename);
        return nvmobj;
    }

    void savenvm_new(NVM_write *t, float *arr, size_t total_size, int num_threads, int parall_iter)
    {
        t->savenvmNew(arr, total_size, num_threads, parall_iter);
    }
}

int main(int argc, char **argv)
{
    return 0;
}