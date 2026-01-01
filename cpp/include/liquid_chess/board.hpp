#pragma once
#include <cstdint>
#include <array>

class BitBoard {
private:
    uint64_t boards[14]; // 6 white pieces, 6 black pieces, empty, occupied
    uint64_t zobrist_key;
    
public:
    // SIMD-optimized move generation
    __m256i generate_moves_simd(Color color) const;
    
    // Magic bitboards for sliding pieces
    uint64_t get_bishop_attacks(int square, uint64_t occupancy) const;
    uint64_t get_rook_attacks(int square, uint64_t occupancy) const;
    
    // Inline functions for critical paths
    __forceinline bool is_attacked(int square, Color color) const {
        // Highly optimized attack detection
    }
};