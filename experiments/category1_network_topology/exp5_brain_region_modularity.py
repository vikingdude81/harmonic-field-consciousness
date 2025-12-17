#!/usr/bin/env python3
"""
Category 1: Network Topology

Experiment 5: Brain Region Modularity Analysis

Tests optimal module count across different brain region configurations:
1. Visual cortex (highly hierarchical, many modules)
2. Prefrontal cortex (integrated, fewer modules)
3. Default Mode Network (3-4 major hubs)
4. Motor cortex (somatotopic organization)
5. Subcortical-cortical interactions
6. Hemispheric organization (left/right)
7. Developmental changes (child → adult → elderly)

Key question: Does optimal modularity differ by brain region/state?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np