import jax
import jax.numpy as jnp
from jax import jit, vmap, random
from flax import linen as nn
from typing import Optional, Tuple
import chex

class LiquidReasoningTransformer(nn.Module):
    """Ultra-fast LRT for chess evaluation"""
    config: dict
    
    def setup(self):
        # Board embedding (optimized for chess)
        self.board_embed = nn.Dense(self.config['hidden_dim'])
        
        # Reasoning token (dynamic working memory)
        self.reasoning_token = self.param('reasoning_token',
                                         random.normal,
                                         (1, self.config['reasoning_dim']))
        
        # Transformer layers with adaptive computation
        self.transformer_layers = [
            AdaptiveTransformerLayer(self.config)
            for _ in range(self.config['num_layers'])
        ]
        
        # Liquid gates
        self.discard_gate = DiscardGate(self.config)
        self.stop_gate = StopGate(self.config)
        self.consistency_head = ConsistencyHead(self.config)
        
        # Evaluation heads
        self.value_head = ValueHead(self.config)
        self.policy_head = PolicyHead(self.config)
        
        # Cache for fast inference
        self._cache = {}
    
    @nn.compact
    def __call__(self, 
                 board_tensor: jnp.ndarray,
                 reasoning_token: Optional[jnp.ndarray] = None,
                 deterministic: bool = True,
                 max_steps: int = 50) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """
        Adaptive reasoning with liquid computation.
        
        Args:
            board_tensor: [64, features] board representation
            reasoning_token: Optional initial reasoning token
            deterministic: Whether to use deterministic gates
            max_steps: Maximum reasoning steps
        
        Returns:
            value: Position evaluation in centipawns
            policy: Move probabilities
            steps: Actual reasoning steps used
        """
        # Encode board
        board_emb = self.board_embed(board_tensor)  # [64, hidden_dim]
        
        # Initialize or use provided reasoning token
        if reasoning_token is None:
            reasoning_token = self.reasoning_token
        
        # Adaptive reasoning loop
        def cond_fn(state):
            token, step, should_continue = state
            return jnp.logical_and(
                step < max_steps,
                should_continue
            )
        
        def body_fn(state):
            token, step, _ = state
            
            # Single transformer pass
            new_token = self.transformer_step(board_emb, token)
            
            # Apply discard gate
            keep_prob = self.discard_gate(token, new_token)
            if deterministic:
                keep = keep_prob > 0.5
            else:
                keep = random.bernoulli(self.make_rng('dropout'), keep_prob)
            
            token = jnp.where(keep, new_token, token)
            
            # Check consistency (legal move filter)
            legal_mask = self.consistency_head(token, board_emb)
            
            # Check if we should stop
            stop_prob = self.stop_gate(token)
            should_continue = stop_prob < 0.95
            
            return token, step + 1, should_continue
        
        # Execute adaptive loop
        final_token, steps, _ = jax.lax.while_loop(
            cond_fn, body_fn, 
            (reasoning_token, 0, True)
        )
        
        # Generate outputs
        value = self.value_head(final_token)
        policy = self.policy_head(final_token, board_emb)
        
        # Apply legal move mask
        policy = policy * self.consistency_head(final_token, board_emb)
        
        return value, policy, steps
    
    def transformer_step(self, board_emb: jnp.ndarray, 
                        reasoning_token: jnp.ndarray) -> jnp.ndarray:
        """Single transformer pass with board-reasoning attention."""
        # Concatenate reasoning token to board
        tokens = jnp.concatenate([
            board_emb, 
            reasoning_token
        ], axis=0)  # [65, hidden_dim]
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        
        # Extract updated reasoning token (last token)
        return tokens[-1:]
    
    @jit
    def evaluate_fast(self, 
                     board_tensor: jnp.ndarray,
                     cache_key: Optional[str] = None) -> float:
        """
        Ultra-fast evaluation with caching and adaptive computation.
        
        This is the main function called by the C++ engine.
        """
        # Check cache first
        if cache_key is not None and cache_key in self._cache:
            value, steps, timestamp = self._cache[cache_key]
            # Cache valid for 1 second
            if (jax.lax.stop_gradient(timestamp) - jax.lax.stop_gradient(jnp.float32(0))) < 1.0:
                return value
        
        # Estimate position complexity
        complexity = self.estimate_complexity(board_tensor)
        
        # Adaptive max steps based on complexity
        max_steps = jnp.clip(
            jnp.int32(complexity * 50),
            2,  # Minimum steps
            50  # Maximum steps
        )
        
        # Run LRT
        value, _, steps = self.__call__(
            board_tensor,
            max_steps=max_steps,
            deterministic=True
        )
        
        # Convert to centipawns
        value_cp = 100 * jnp.tanh(value[0, 0])
        
        # Cache result
        if cache_key is not None:
            self._cache[cache_key] = (
                value_cp, 
                steps, 
                jax.lax.stop_gradient(jnp.float32(0))  # Current time
            )
            # Limit cache size
            if len(self._cache) > 10000:
                self._cache.pop(next(iter(self._cache)))
        
        return value_cp
    
    def estimate_complexity(self, board_tensor: jnp.ndarray) -> float:
        """Estimate position complexity to determine reasoning depth."""
        # Simple heuristics for speed
        material_count = jnp.sum(board_tensor[:, :12])  # Piece count
        piece_density = jnp.mean(board_tensor[:, 12:])  # Positional features
        
        # More pieces + more centralization = more complex
        complexity = (
            0.3 * material_count +
            0.7 * piece_density
        )
        
        return jnp.clip(complexity, 0.0, 1.0)

class AdaptiveTransformerLayer(nn.Module):
    """Transformer layer with adaptive computation."""
    config: dict
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Self-attention
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.config['num_heads'],
            dropout_rate=self.config['dropout_rate']
        )(x, x)
        
        x = x + attn_out
        x = nn.LayerNorm()(x)
        
        # Feed-forward with adaptive width
        ff_dim = self.config['ff_dim']
        
        # Dynamic width based on input "confusion"
        confusion = jnp.std(x, axis=-1).mean()
        adaptive_width = jnp.int32(
            ff_dim * (0.5 + 0.5 * jnp.tanh(confusion * 10))
        )
        
        # Sparse FFN
        ff_out = nn.Dense(adaptive_width)(x)
        ff_out = nn.relu(ff_out)
        ff_out = nn.Dense(x.shape[-1])(ff_out)
        
        x = x + ff_out
        x = nn.LayerNorm()(x)
        
        return x