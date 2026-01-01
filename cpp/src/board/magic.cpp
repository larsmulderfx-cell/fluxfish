#include "liquid_chess/bitboard.hpp"
#include <array>
#include <algorithm>

namespace LiquidChess {

// Pre-calculated magic numbers (from Stockfish)
const Bitboard BishopMagics[64] = {
    0x007fbfbfbfbfbfffu, 0x0000a060401007fcu, 0x0001004008020000u, /* ... */
};

const Bitboard RookMagics[64] = {
    0x0080001020400080u, 0x0040001000200040u, 0x0080081000200080u, /* ... */
};

// Initialize attack tables at compile time
template<typename Function>
void init_slider_attacks(Function attack, Bitboard* table[], 
                         const Bitboard magics[], Bitboard masks[]) {
    
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        // Allocate table for this square
        Bitboard* attacks = table[s];
        Bitboard mask = masks[s];
        int shift = 64 - popcount(mask);
        
        // Generate all possible occupancies
        Bitboard occ = 0;
        do {
            Bitboard idx = (occ * magics[s]) >> shift;
            attacks[idx] = attack(s, occ);
            occ = (occ - mask) & mask;
        } while (occ);
    }
}

// Initialize all attack tables (called once at startup)
void init_magics() {
    static std::once_flag flag;
    std::call_once(flag, [](){
        // Allocate memory for attack tables
        for (Square s = SQ_A1; s <= SQ_H8; ++s) {
            BishopAttacks[s] = new Bitboard[1 << BishopShift[s]];
            RookAttacks[s] = new Bitboard[1 << RookShift[s]];
        }
        
        // Initialize tables
        init_slider_attacks(bishop_attacks, BishopAttacks, 
                           BishopMagics, BishopMask);
        init_slider_attacks(rook_attacks, RookAttacks, 
                           RookMagics, RookMask);
    });
}

}