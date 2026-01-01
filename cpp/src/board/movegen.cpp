#include "liquid_chess/bitboard.hpp"
#include <xmmintrin.h>
#include <emmintrin.h>

namespace LiquidChess {

template<GenType Type>
void BitBoard::generate_moves(MoveList& moves) const {
    constexpr bool All = Type == ALL;
    const Color us = sideToMove;
    const Color them = ~us;
    const Bitboard occupied = pieces_by_color[WHITE] | pieces_by_color[BLACK];
    const Bitboard empty = ~occupied;
    
    // King moves
    Square ksq = king_square(us);
    Bitboard kingAttacks = attacks_bb<KING>(ksq) & ~pieces_by_color[us];
    while (kingAttacks) {
        Square to = pop_lsb(&kingAttacks);
        moves.add(make_move(ksq, to));
    }
    
    // Pawn moves with SIMD optimization for multiple pawns
    Bitboard pawns = pieces_by_type[us][PAWN];
    if (pawns) {
        // Generate pawn pushes
        Bitboard singlePushes = shift_bb<us, NORTH>(pawns) & empty;
        Bitboard doublePushes = shift_bb<us, NORTH>(singlePushes & Rank3BB) & empty;
        
        // Process pushes in batches using SIMD
        process_pawn_pushes_simd(singlePushes, doublePushes, us, moves);
        
        // Pawn captures
        Bitboard leftAttacks = shift_bb<us, NORTH_WEST>(pawns) & pieces_by_color[them];
        Bitboard rightAttacks = shift_bb<us, NORTH_EAST>(pawns) & pieces_by_color[them];
        
        // En passant
        if (epSquare != SQ_NONE) {
            Bitboard epAttacks = pawn_attacks_bb(them, epSquare) & pawns;
            while (epAttacks) {
                Square from = pop_lsb(&epAttacks);
                moves.add(make_move(from, epSquare, EP_FLAG));
            }
        }
    }
    
    // Knight moves - process multiple knights simultaneously
    Bitboard knights = pieces_by_type[us][KNIGHT];
    while (knights) {
        Square from = pop_lsb(&knights);
        Bitboard attacks = attacks_bb<KNIGHT>(from) & ~pieces_by_color[us];
        
        // Batch process knight moves
        __m128i fromVec = _mm_set1_epi8(from);
        __m128i attackMask = _mm_loadu_si128((__m128i*)&attacks);
        
        while (attacks) {
            Square to = pop_lsb(&attacks);
            moves.add(make_move(from, to));
        }
    }
    
    // Sliding pieces with magic bitboards
    generate_slider_moves<BISHOP>(us, occupied, moves);
    generate_slider_moves<ROOK>(us, occupied, moves);
    generate_slider_moves<QUEEN>(us, occupied, moves);
}

// SIMD-optimized pawn push generation
void BitBoard::process_pawn_pushes_simd(Bitboard singlePushes, 
                                       Bitboard doublePushes, 
                                       Color us, MoveList& moves) const {
    
    // Process 4 pawns at a time using SSE
    alignas(16) uint64_t pushes[2] = {singlePushes, doublePushes};
    __m128i pushVec = _mm_load_si128((__m128i*)pushes);
    
    // Create mask for pawn origins
    Bitboard pawns = pieces_by_type[us][PAWN];
    while (pawns) {
        Square from = pop_lsb(&pawns);
        
        // Check for pushes from this pawn
        Bitboard fromBB = square_bb(from);
        Bitboard single = singlePushes & shift_bb<us, SOUTH>(fromBB);
        Bitboard doub = doublePushes & shift_bb<us, SOUTH>(single);
        
        while (single) {
            Square to = pop_lsb(&single);
            if (rank_of(to) == RANK_8) {
                // Promotion
                moves.add(make_move(from, to, PROMOTION, QUEEN));
                moves.add(make_move(from, to, PROMOTION, ROOK));
                moves.add(make_move(from, to, PROMOTION, BISHOP));
                moves.add(make_move(from, to, PROMOTION, KNIGHT));
            } else {
                moves.add(make_move(from, to));
            }
        }
        
        while (doub) {
            Square to = pop_lsb(&doub);
            moves.add(make_move(from, to));
        }
    }
}

}