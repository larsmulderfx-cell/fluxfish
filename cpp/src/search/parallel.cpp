#include "liquid_chess/search.hpp"
#include <atomic>
#include <thread>
#include <vector>
#include <deque>
#include <mutex>

namespace LiquidChess {

class WorkStealingQueue {
private:
    struct alignas(64) Task {
        Position pos;
        int depth;
        int alpha;
        int beta;
        MoveList pv;
    };
    
    std::deque<Task> queue;
    mutable std::mutex mutex;
    
public:
    void push_front(const Task& task) {
        std::lock_guard lock(mutex);
        queue.push_front(task);
    }
    
    bool pop_front(Task& task) {
        std::lock_guard lock(mutex);
        if (queue.empty()) return false;
        task = queue.front();
        queue.pop_front();
        return true;
    }
    
    bool pop_back(Task& task) {
        std::lock_guard lock(mutex);
        if (queue.empty()) return false;
        task = queue.back();
        queue.pop_back();
        return true;
    }
    
    size_t size() const {
        std::lock_guard lock(mutex);
        return queue.size();
    }
};

class ThreadPool {
private:
    std::vector<std::thread> threads;
    std::vector<WorkStealingQueue> queues;
    std::atomic<bool> stop;
    std::atomic<int> activeWorkers;
    
    // Thread-local data
    struct ThreadData {
        Position pos;
        SearchStack* ss;
        uint64_t nodes;
        int threadId;
    };
    
    std::vector<ThreadData> threadData;
    
public:
    ThreadPool(int numThreads) : stop(false), activeWorkers(0) {
        queues.resize(numThreads);
        threadData.resize(numThreads);
        
        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back([this, i]() {
                worker_loop(i);
            });
            
            // Pin threads to cores for better cache locality
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i % std::thread::hardware_concurrency(), &cpuset);
            pthread_setaffinity_np(threads[i].native_handle(),
                                  sizeof(cpu_set_t), &cpuset);
        }
    }
    
    ~ThreadPool() {
        stop = true;
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
    }
    
    void worker_loop(int threadId) {
        ThreadData& data = threadData[threadId];
        data.threadId = threadId;
        
        while (!stop) {
            // Try to get work from own queue first
            Task task;
            if (queues[threadId].pop_front(task)) {
                activeWorkers++;
                search_task(task, data);
                activeWorkers--;
                continue;
            }
            
            // Try to steal work from other threads
            for (int i = 0; i < queues.size(); ++i) {
                if (i == threadId) continue;
                
                if (queues[i].pop_back(task)) {
                    activeWorkers++;
                    search_task(task, data);
                    activeWorkers--;
                    break;
                }
            }
            
            // No work available, yield
            std::this_thread::yield();
        }
    }
    
    // Main search entry point
    Move search(const Position& rootPos, int maxDepth, 
                TimeManager& tm, int numThreads) {
        
        // Initialize thread data
        for (int i = 0; i < numThreads; ++i) {
            threadData[i].pos = rootPos;
            threadData[i].nodes = 0;
            threadData[i].ss = new SearchStack[MAX_PLY + 10];
        }
        
        // Create root task
        Task rootTask;
        rootTask.pos = rootPos;
        rootTask.depth = 1;
        rootTask.alpha = -INFINITE;
        rootTask.beta = INFINITE;
        
        queues[0].push_front(rootTask);
        
        // Iterative deepening
        Move bestMove = MOVE_NONE;
        for (int depth = 1; depth <= maxDepth && !tm.timeout(); ++depth) {
            
            // Update task depth
            rootTask.depth = depth;
            queues[0].push_front(rootTask);
            
            // Wait for completion
            while (activeWorkers > 0 && !tm.timeout()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            
            // Collect results
            // ... (result collection logic)
        }
        
        return bestMove;
    }
};

}