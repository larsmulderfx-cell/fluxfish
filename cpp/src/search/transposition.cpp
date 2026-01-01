#include "liquid_chess/search.hpp"
#include <immintrin.h>

namespace LiquidChess {

class TranspositionTable {
private:
    struct Cluster {
        TTEntry entries[3]; // 3 entries per cluster for better utilization
        uint8_t padding[64 - sizeof(TTEntry) * 3]; // Pad to cache line
    };
    
    static_assert(sizeof(Cluster) == 64, "Cluster must be cache line sized");
    
    Cluster* table;
    size_t clusterCount;
    std::atomic<uint8_t> generation;
    
public:
    TranspositionTable(size_t mbSize) {
        // Calculate size as power of two clusters
        clusterCount = 1;
        while ((clusterCount * sizeof(Cluster)) < mbSize * 1024 * 1024)
            clusterCount <<= 1;
        clusterCount >>= 1;
        
        // Allocate aligned memory
        table = (Cluster*)aligned_alloc(64, clusterCount * sizeof(Cluster));
        memset(table, 0, clusterCount * sizeof(Cluster));
    }
    
    ~TranspositionTable() {
        free(table);
    }
    
    // Prefetch the cluster for a given key
    __forceinline void prefetch(uint64_t key) const {
        size_t idx = (key >> 48) & (clusterCount - 1);
        _mm_prefetch((const char*)&table[idx], _MM_HINT_T0);
    }
    
    // Probe the table with aggressive prefetching
    TTEntry* probe(uint64_t key, bool& found) {
        size_t idx = (key >> 48) & (clusterCount - 1);
        Cluster& cluster = table[idx];
        
        // Prefetch next likely cluster
        _mm_prefetch((const char*)&table[(idx + 1) & (clusterCount - 1)], _MM_HINT_T1);
        
        uint16_t key16 = key >> 48;
        
        // Check all entries in the cluster
        for (int i = 0; i < 3; ++i) {
            if (cluster.entries[i].key16 == key16) {
                found = true;
                cluster.entries[i].gen = generation;
                return &cluster.entries[i];
            }
        }
        
        found = false;
        
        // Find entry to replace (LFU strategy)
        TTEntry* replace = &cluster.entries[0];
        for (int i = 1; i < 3; ++i) {
            // Replace based on depth and generation age
            if (cluster.entries[i].depth8 - 2 * (generation - cluster.entries[i].gen) 
                < replace->depth8 - 2 * (generation - replace->gen)) {
                replace = &cluster.entries[i];
            }
        }
        
        return replace;
    }
    
    // Store an entry
    void store(uint64_t key, int16_t value, uint8_t depth, 
               uint8_t bound, Move bestMove, uint8_t staticEval) {
        
        size_t idx = (key >> 48) & (clusterCount - 1);
        Cluster& cluster = table[idx];
        uint16_t key16 = key >> 48;
        
        TTEntry* entry = nullptr;
        
        // Try to replace an existing entry with same key
        for (int i = 0; i < 3; ++i) {
            if (cluster.entries[i].key16 == key16) {
                entry = &cluster.entries[i];
                break;
            }
        }
        
        // Otherwise replace the least valuable entry
        if (!entry) {
            entry = &cluster.entries[0];
            for (int i = 1; i < 3; ++i) {
                if (cluster.entries[i].depth8 - 2 * (generation - cluster.entries[i].gen) 
                    < entry->depth8 - 2 * (generation - entry->gen)) {
                    entry = &cluster.entries[i];
                }
            }
        }
        
        // Don't replace deeper entries with shallower ones
        if (entry->key16 == key16 && entry->depth8 > depth && bound == EXACT)
            return;
        
        // Store the entry
        entry->key16 = key16;
        entry->move16 = bestMove;
        entry->value16 = value;
        entry->eval16 = staticEval;
        entry->gen = generation;
        entry->depth8 = depth;
        entry->bound8 = bound;
    }
    
    void new_search() {
        generation++;
    }
};

}