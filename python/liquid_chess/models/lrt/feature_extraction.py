"""
Chess-specific feature extraction for enhanced board encoding
"""

import jax.numpy as jnp
import chess
from typing import Tuple, Dict


class ChessFeatureExtractor:
    """Extract advanced chess features for board encoding"""
    
    @staticmethod
    def extract_attack_maps(board: chess.Board) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Extract attack maps for both colors"""
        white_attacks = jnp.zeros(64, dtype=jnp.float32)
        black_attacks = jnp.zeros(64, dtype=jnp.float32)
        
        for square in chess.SQUARES:
            # White attacks
            if board.is_attacked_by(chess.WHITE, square):
                white_attacks = white_attacks.at[square].set(1.0)
            # Black attacks
            if board.is_attacked_by(chess.BLACK, square):
                black_attacks = black_attacks.at[square].set(1.0)
        
        return white_attacks, black_attacks
    
    @staticmethod
    def extract_pin_detection(board: chess.Board) -> jnp.ndarray:
        """Detect pinned pieces using python-chess built-in methods"""
        pins = jnp.zeros(64, dtype=jnp.float32)
        
        # Use python-chess's built-in pin detection
        for color in [chess.WHITE, chess.BLACK]:
            king_sq = board.king(color)
            if king_sq is None:
                continue
                
            # Check each piece for being pinned
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color:
                    # Check if piece is pinned to king
                    if board.is_pinned(color, square):
                        pins = pins.at[square].set(1.0)
        
        return pins
    
    @staticmethod
    def extract_king_safety(board: chess.Board) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Extract king safety zones (3x3 area around king, clipped to board edges)"""
        white_safety = jnp.zeros(64, dtype=jnp.float32)
        black_safety = jnp.zeros(64, dtype=jnp.float32)
        
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        def get_safety_zone(king_sq):
            if king_sq is None:
                return jnp.zeros(64, dtype=jnp.float32)
            
            zone = jnp.zeros(64, dtype=jnp.float32)
            king_file = chess.square_file(king_sq)
            king_rank = chess.square_rank(king_sq)
            
            # 3x3 area around king (or smaller on edges)
            for df in [-1, 0, 1]:
                for dr in [-1, 0, 1]:
                    f = king_file + df
                    r = king_rank + dr
                    if 0 <= f < 8 and 0 <= r < 8:
                        sq = chess.square(f, r)
                        zone = zone.at[sq].set(1.0)
            
            return zone
        
        white_safety = get_safety_zone(white_king)
        black_safety = get_safety_zone(black_king)
        
        return white_safety, black_safety
    
    @staticmethod
    def extract_pawn_structure(board: chess.Board) -> Dict[str, jnp.ndarray]:
        """Extract pawn structure features"""
        features = {
            'passed_pawns': jnp.zeros(64, dtype=jnp.float32),
            'isolated_pawns': jnp.zeros(64, dtype=jnp.float32),
            'doubled_pawns': jnp.zeros(64, dtype=jnp.float32),
        }
        
        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            
            for pawn_sq in pawns:
                file = chess.square_file(pawn_sq)
                rank = chess.square_rank(pawn_sq)
                
                # Check for passed pawn
                is_passed = True
                enemy_pawns = board.pieces(chess.PAWN, not color)
                
                for enemy_sq in enemy_pawns:
                    enemy_file = chess.square_file(enemy_sq)
                    enemy_rank = chess.square_rank(enemy_sq)
                    
                    # Check if enemy pawn can block
                    if abs(enemy_file - file) <= 1:
                        if color == chess.WHITE and enemy_rank > rank:
                            is_passed = False
                            break
                        elif color == chess.BLACK and enemy_rank < rank:
                            is_passed = False
                            break
                
                if is_passed:
                    features['passed_pawns'] = features['passed_pawns'].at[pawn_sq].set(1.0)
                
                # Check for isolated pawn
                has_neighbor = False
                for adj_file in [file - 1, file + 1]:
                    if 0 <= adj_file < 8:
                        for r in range(8):
                            sq = chess.square(adj_file, r)
                            piece = board.piece_at(sq)
                            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                                has_neighbor = True
                                break
                
                if not has_neighbor:
                    features['isolated_pawns'] = features['isolated_pawns'].at[pawn_sq].set(1.0)
                
                # Check for doubled pawns
                for r in range(8):
                    if r != rank:
                        sq = chess.square(file, r)
                        piece = board.piece_at(sq)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            features['doubled_pawns'] = features['doubled_pawns'].at[pawn_sq].set(1.0)
                            break
        
        return features
    
    @staticmethod
    def extract_material_imbalance(board: chess.Board) -> jnp.ndarray:
        """Extract material imbalance per square (for context)"""
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0,
        }
        
        white_material = 0.0
        black_material = 0.0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Normalized imbalance (positive = white advantage, negative = black advantage)
        max_material = 39.0  # Maximum possible material (excluding kings)
        imbalance = (white_material - black_material) / max_material
        
        # Broadcast to all squares
        return jnp.full(64, imbalance, dtype=jnp.float32)


def board_to_enhanced_input(board: chess.Board) -> Dict[str, jnp.ndarray]:
    """
    Convert python-chess Board to enhanced input format (highly optimized)
    
    Args:
        board: chess.Board object
    
    Returns:
        Dictionary with all enhanced features
    """
    # Basic board state
    pieces = jnp.zeros((8, 8), dtype=jnp.int8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Map piece to index (1-12)
            idx = (piece.piece_type - 1) * 2 + (1 if piece.color == chess.WHITE else 2)
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            pieces = pieces.at[rank, file].set(idx)
    
    # Fast attack maps (vectorized)
    white_attacks = jnp.zeros(64, dtype=jnp.float32)
    black_attacks = jnp.zeros(64, dtype=jnp.float32)
    
    # Only attack maps for performance
    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            white_attacks = white_attacks.at[square].set(1.0)
        if board.is_attacked_by(chess.BLACK, square):
            black_attacks = black_attacks.at[square].set(1.0)
    
    # Pin detection (use the proper extractor method)
    extractor = ChessFeatureExtractor()
    pins = extractor.extract_pin_detection(board)
    
    # Minimal other features for performance
    king_safety_white = jnp.zeros(64, dtype=jnp.float32)
    king_safety_black = jnp.zeros(64, dtype=jnp.float32)
    
    white_king = board.king(chess.WHITE)
    if white_king is not None:
        kf, kr = chess.square_file(white_king), chess.square_rank(white_king)
        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                f, r = kf + df, kr + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    king_safety_white = king_safety_white.at[chess.square(f, r)].set(1.0)
    
    black_king = board.king(chess.BLACK)
    if black_king is not None:
        kf, kr = chess.square_file(black_king), chess.square_rank(black_king)
        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                f, r = kf + df, kr + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    king_safety_black = king_safety_black.at[chess.square(f, r)].set(1.0)
    
    # Simple material calculation
    piece_values = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
    white_material = sum(len(board.pieces(pt, chess.WHITE)) * piece_values.get(pt, 0) for pt in range(1, 7))
    black_material = sum(len(board.pieces(pt, chess.BLACK)) * piece_values.get(pt, 0) for pt in range(1, 7))
    
    max_material = 39.0
    imbalance = (white_material - black_material) / max_material
    material_imbalance = jnp.full(64, imbalance, dtype=jnp.float32)
    
    # Minimal features for performance
    passed_pawns = jnp.zeros(64, dtype=jnp.float32)
    isolated_pawns = jnp.zeros(64, dtype=jnp.float32)
    doubled_pawns = jnp.zeros(64, dtype=jnp.float32)
    
    return {
        'pieces': pieces,
        'turn': jnp.array(board.turn, dtype=jnp.bool_),
        'castling': jnp.array([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK)
        ], dtype=jnp.bool_),
        'ep_square': jnp.array(
            board.ep_square if board.ep_square else -1,
            dtype=jnp.int8
        ),
        'white_attacks': white_attacks,
        'black_attacks': black_attacks,
        'pins': pins,
        'king_safety_white': king_safety_white,
        'king_safety_black': king_safety_black,
        'passed_pawns': passed_pawns,
        'isolated_pawns': isolated_pawns,
        'doubled_pawns': doubled_pawns,
        'material_imbalance': material_imbalance,
    }