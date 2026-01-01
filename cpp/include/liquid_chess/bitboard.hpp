#pragma once
#include <cstdint>
#include <array>
#include <immintrin.h>

namespace LiquidChess {

// Compile-time constants
constexpr int SQUARE_NB = 64;
constexpr int PIECE_NB = 12;
constexpr int COLOR_NB = 2;

// Bitboard type
using Bitboard = uint64_t;

// Magic bitboard constants (pre-calculated)
extern const Bitboard BishopMagics[64];
extern const Bitboard RookMagics[64];
extern const Bitboard* BishopAttacks[64];
extern const Bitboard* RookAttacks[64];
extern const Bitboard BetweenBB[64][64];
extern const Bitboard LineBB[64][64];
extern const Bitboard PseudoAttacks[PIECE_NB][64];

class BitBoard {
private:
    // Board representation - 12 piece bitboards + occupied squares
    Bitboard pieces[PIECE_NB];
    Bitboard occupied[COLOR_NB];
    Bitboard empty;
    
    // Zobrist hashing
    uint64_t key;
    uint64_t pawnKey;
    uint64_t materialKey;
    
    // Piece lists for faster iteration
    uint8_t pieceList[PIECE_NB][10]; // Max 10 of each piece
    uint8_t pieceCount[PIECE_NB];
    
    // Cached attack tables
    mutable Bitboard pawnAttacks[COLOR_NB][64];
    mutable bool attackTableDirty;
    
public:
    BitBoard();
    
    // Core operations (force inline for performance)
    __forceinline void set_piece(Square sq, Piece pc) {
        Bitboard bb = square_bb(sq);
        pieces[pc] ^= bb;
        occupied[color_of(pc)] ^= bb;
        empty ^= bb;
        key ^= Zobrist::psq[pc][sq];
        
        // Update piece list
        pieceList[pc][pieceCount[pc]++] = sq;
    }
    
    __forceinline void remove_piece(Square sq, Piece pc) {
        Bitboard bb = square_bb(sq);
        pieces[pc] ^= bb;
        occupied[color_of(pc)] ^= bb;
        empty ^= bb;
        key ^= Zobrist::psq[pc][sq];
        
        // Update piece list
        for (int i = 0; i < pieceCount[pc]; ++i) {
            if (pieceList[pc][i] == sq) {
                pieceList[pc][i] = pieceList[pc][--pieceCount[pc]];
                break;
            }
        }
    }
    
    // Attack generation with magic bitboards
    __forceinline Bitboard attacks_bb(PieceType pt, Square s, Bitboard occupied) const {
        switch (pt) {
            case BISHOP: return bishop_attacks_bb(s, occupied);
            case ROOK:   return rook_attacks_bb(s, occupied);
            case QUEEN:  return bishop_attacks_bb(s, occupied) | 
                              rook_attacks_bb(s, occupied);
            default:     return PseudoAttacks[pt][s];
        }
    }
    
    __forceinline Bitboard bishop_attacks_bb(Square s, Bitboard occupied) const {
        return BishopAttacks[s][((occupied & BishopMask[s]) * BishopMagics[s]) >> BishopShift[s]];
    }
    
    __forceinline Bitboard rook_attacks_bb(Square s, Bitboard occupied) const {
        return RookAttacks[s][((occupied & RookMask[s]) * RookMagics[s]) >> RookShift[s]];
    }
    
    // SIMD-optimized multiple square attack detection
    __m256i attacks_to_multiple_avx2(Square* squares, int count, Color c) const;
    
    // Move generation using pre-calculated attack tables
    template<GenType Type>
    void generate_moves(MoveList& moves) const;
    
    // Fast evaluation (material + PST)
    __forceinline int simple_eval() const {
        int score = 0;
        // Material counting with piece-square tables
        for (Piece pc = W_PAWN; pc <= B_KING; ++pc) {
            Bitboard bb = pieces[pc];
            while (bb) {
                Square s = pop_lsb(&bb);
                score += PieceValue[pc] + PSQT[pc][s];
            }
        }
        return score;
    }
    
    // Bulk counting using population count
    __forceinline int count_material(Color c) const {
        constexpr int values[] = {100, 300, 300, 500, 900, 0};
        int total = 0;
        for (PieceType pt = PAWN; pt <= KING; ++pt) {
            total += values[pt] * popcount(pieces[make_piece(c, pt)]);
        }
        return total;
    }
};
}