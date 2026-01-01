#pragma once

struct alignas(64) SearchNode { // Cache-line aligned
    uint64_t key;
    int16_t value;
    uint8_t depth;
    uint8_t node_type; // EXACT, LOWER_BOUND, UPPER_BOUND
    uint16_t best_move;
    uint8_t generation;
    
    // Padding to 64 bytes
    uint8_t padding[64 - 20];
};

class TranspositionTable {
private:
    SearchNode* table;
    size_t size;
    
public:
    // Prefetch-friendly access pattern
    void prefetch(uint64_t key) const {
        __builtin_prefetch(&table[key & (size - 1)]);
    }
};