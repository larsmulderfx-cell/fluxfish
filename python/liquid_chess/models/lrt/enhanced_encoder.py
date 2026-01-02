"""
Enhanced Board Encoder with Chess-Specific Features
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Dict


class EnhancedChessBoardEncoder(nn.Module):
    """Enhanced board encoder with chess-specific features"""
    features: int = 512
    
    @nn.compact
    def __call__(self, board_state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Encode board with sophisticated chess features
        
        Args:
            board_state: Dictionary containing board features
        
        Returns:
            encoded: [64, features] encoded board representation
        """
        
        # 1. Basic piece embeddings
        pieces_flat = board_state['pieces'].flatten()
        piece_emb = nn.Embed(13, self.features // 8)(pieces_flat)
        piece_emb = piece_emb.reshape(64, -1)
        
        # 2. Positional features
        ranks = jnp.arange(8).repeat(8).reshape(64, 1) / 7.0
        files = jnp.tile(jnp.arange(8), 8).reshape(64, 1) / 7.0
        
        center_dist = (jnp.abs(ranks - 3.5/7.0) + jnp.abs(files - 3.5/7.0))
        back_rank = jnp.logical_or(ranks == 0, ranks == 1.0).astype(jnp.float32).reshape(64, 1)
        diag1 = (ranks == files).astype(jnp.float32).reshape(64, 1)
        diag2 = (ranks == (1.0 - files)).astype(jnp.float32).reshape(64, 1)
        
        square_features = jnp.concatenate([
            ranks, files, center_dist, back_rank, diag1, diag2
        ], axis=-1)
        
        pos_emb = nn.Dense(self.features // 8)(square_features)
        
        # 3. Attack maps
        white_attacks = board_state.get('white_attacks', jnp.zeros(64, dtype=jnp.float32))
        black_attacks = board_state.get('black_attacks', jnp.zeros(64, dtype=jnp.float32))
        
        attack_features = jnp.stack([white_attacks, black_attacks], axis=-1)
        attack_emb = nn.Dense(self.features // 16)(attack_features)
        
        # 4. Pin detection
        pins = board_state.get('pins', jnp.zeros(64, dtype=jnp.float32)).reshape(64, 1)
        pin_emb = nn.Dense(self.features // 16)(pins)
        
        # 5. King safety zones
        king_safety_white = board_state.get('king_safety_white', jnp.zeros(64, dtype=jnp.float32))
        king_safety_black = board_state.get('king_safety_black', jnp.zeros(64, dtype=jnp.float32))
        
        king_safety = jnp.stack([king_safety_white, king_safety_black], axis=-1)
        king_safety_emb = nn.Dense(self.features // 16)(king_safety)
        
        # 6. Pawn structure features
        passed_pawns = board_state.get('passed_pawns', jnp.zeros(64, dtype=jnp.float32))
        isolated_pawns = board_state.get('isolated_pawns', jnp.zeros(64, dtype=jnp.float32))
        doubled_pawns = board_state.get('doubled_pawns', jnp.zeros(64, dtype=jnp.float32))
        
        pawn_structure = jnp.stack([passed_pawns, isolated_pawns, doubled_pawns], axis=-1)
        pawn_emb = nn.Dense(self.features // 16)(pawn_structure)
        
        # 7. Material imbalance
        material_imb = board_state.get('material_imbalance', jnp.zeros(64, dtype=jnp.float32))
        material_emb = nn.Dense(self.features // 16)(material_imb.reshape(64, 1))
        
        # 8. Global metadata
        turn_feat = board_state['turn'].astype(jnp.float32).reshape(1, 1)
        castling_feat = board_state['castling'].astype(jnp.float32).reshape(1, 4)
        ep_feat = (board_state['ep_square'] >= 0).astype(jnp.float32).reshape(1, 1)
        
        meta = jnp.concatenate([turn_feat, castling_feat, ep_feat], axis=-1)
        meta_emb = nn.Dense(self.features // 8)(meta)
        meta_emb = jnp.tile(meta_emb, (64, 1))
        
        # 9. Combine all features
        combined = jnp.concatenate([
            piece_emb,
            pos_emb,
            attack_emb,
            pin_emb,
            king_safety_emb,
            pawn_emb,
            material_emb,
            meta_emb
        ], axis=-1)
        
        # 10. Final projection with residual
        encoded = nn.Dense(self.features)(combined)
        encoded = nn.LayerNorm()(encoded)
        
        # Add residual from piece embeddings
        piece_proj = nn.Dense(self.features)(piece_emb)
        encoded = encoded + 0.1 * piece_proj
        
        encoded = nn.LayerNorm()(encoded)
        
        return encoded