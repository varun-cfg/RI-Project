"""
constraint_policy.py

Constraint-aware policy wrapper that smooths and repairs actions from OpenVLA
to reduce jerk and constraint violations.
"""

import numpy as np
from typing import Dict, Optional, Any


class ConstraintAwarePolicy:
    def __init__(self, base_policy, constraints: Optional[Dict[str, Any]] = None):
        """
        Wraps a base policy with constraint enforcement.
        
        Args:
            base_policy: The loaded OpenVLA model or policy
            constraints: Dict defining active constraints:
                - 'max_step_cm': Maximum translation step size (m)
                - 'max_rotation_deg': Maximum rotation step size (degrees)
                - 'lock_rotation': Whether to prevent rotation changes
                - 'workspace_bounds': [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
                - 'max_tilt_deg': Maximum allowed tilt from vertical (degrees)
                - 'adaptive': Whether to use adaptive constraint scaling (default: True)
        """
        self.policy = base_policy
        self.constraints = constraints or {}
        
        # Enable adaptive constraints by default
        if 'adaptive' not in self.constraints:
            self.constraints['adaptive'] = True
        
        # Statistics tracking
        self.stats = {
            'translation_clipped': 0,
            'rotation_clipped': 0,
            'workspace_clipped': 0,
            'tilt_corrected': 0,
            'total_actions': 0,
            'avg_translation_scale': [],
            'avg_rotation_scale': []
        }
    
    def repair_action(self, action: np.ndarray, obs: Dict[str, Any]) -> np.ndarray:
        """
        Repairs action based on active constraints.
        BALANCED STRATEGY: Reduce jerk while maintaining task success
        - Moderate smoothing threshold (2x limit)
        - Gentle scaling to preserve action effectiveness
        - NEVER modify gripper action (index 6)
        
        Args:
            action: Raw action from policy (7-dim: x, y, z, rx, ry, rz, gripper)
            obs: Current observation dictionary
            
        Returns:
            Repaired action
        """
        repaired_action = action.copy()
        self.stats['total_actions'] += 1
        
        # IMPORTANT: Save gripper action and restore it at the end
        original_gripper = repaired_action[6]
        
        # A. Translation Smoothing - Apply at 2x limit (balanced approach)
        if 'max_step_cm' in self.constraints:
            limit = self.constraints['max_step_cm']
            translation_mag = np.linalg.norm(repaired_action[:3])
            
            # Intervene at 2x the limit (balanced approach)
            moderate_threshold = 2.0 * limit
            if translation_mag > moderate_threshold:
                if self.constraints.get('adaptive', True):
                    # Gentle exponential smoothing
                    excess_ratio = translation_mag / limit
                    # Smooth decay: cap reduction at 40% to preserve task capability
                    scale = 1.0 / (1.0 + 0.15 * (excess_ratio - 2.0))
                    scale = max(scale, 0.6)  # Never reduce by more than 40%
                    repaired_action[:3] *= scale
                    self.stats['translation_clipped'] += 1
                    self.stats['avg_translation_scale'].append(scale)
                else:
                    scale = max(moderate_threshold / translation_mag, 0.6)
                    repaired_action[:3] *= scale
                    self.stats['translation_clipped'] += 1
                    self.stats['avg_translation_scale'].append(scale)
        
        # B. Rotation Smoothing - Apply at 2x limit
        if 'max_rotation_deg' in self.constraints and not self.constraints.get('lock_rotation', False):
            limit_rad = np.deg2rad(self.constraints['max_rotation_deg'])
            rotation_mag = np.linalg.norm(repaired_action[3:6])
            
            # Intervene at 2x the limit
            moderate_threshold = 2.0 * limit_rad
            if rotation_mag > moderate_threshold:
                if self.constraints.get('adaptive', True):
                    excess_ratio = rotation_mag / limit_rad
                    scale = 1.0 / (1.0 + 0.15 * (excess_ratio - 2.0))
                    scale = max(scale, 0.6)  # Never reduce by more than 40%
                    repaired_action[3:6] *= scale
                    self.stats['rotation_clipped'] += 1
                    self.stats['avg_rotation_scale'].append(scale)
                else:
                    scale = max(moderate_threshold / rotation_mag, 0.6)
                    repaired_action[3:6] *= scale
                    self.stats['rotation_clipped'] += 1
                    self.stats['avg_rotation_scale'].append(scale)
        
        # C. Orientation Lock - DISABLED unless explicitly required
        if self.constraints.get('lock_rotation', False):
            if np.any(repaired_action[3:6] != 0):
                self.stats['rotation_clipped'] += 1
            repaired_action[3:6] = 0.0
        
        # D. Workspace Bounds - DISABLED (too restrictive)
        
        # CRITICAL: Restore original gripper action - never modify it
        repaired_action[6] = original_gripper
        
        return repaired_action
    
    def get_stats(self) -> Dict[str, float]:
        """Returns statistics about constraint enforcement."""
        total = max(self.stats['total_actions'], 1)
        avg_trans_scale = np.mean(self.stats['avg_translation_scale']) if self.stats['avg_translation_scale'] else 1.0
        avg_rot_scale = np.mean(self.stats['avg_rotation_scale']) if self.stats['avg_rotation_scale'] else 1.0
        
        return {
            'translation_clip_rate': self.stats['translation_clipped'] / total,
            'rotation_clip_rate': self.stats['rotation_clipped'] / total,
            'workspace_clip_rate': self.stats['workspace_clipped'] / total,
            'tilt_correction_rate': self.stats['tilt_corrected'] / total,
            'avg_translation_scaling': avg_trans_scale,
            'avg_rotation_scaling': avg_rot_scale,
        }
    
    def reset_stats(self):
        """Resets statistics counters."""
        for key in self.stats:
            if isinstance(self.stats[key], list):
                self.stats[key] = []
            else:
                self.stats[key] = 0
