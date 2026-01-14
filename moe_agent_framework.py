#!/usr/bin/env python3
"""
Multi-Expert Agent Framework with LoRA Adapter Routing

This framework enables dynamic expert switching for different domains:
- Trading/Finance
- Coding/Software
- Consciousness Research
- Game Development
- Agent Orchestration

Architecture:
- Base model: Qwen2.5-32B (loaded once)
- Expert adapters: LoRA weights for each domain
- Smart router: Selects appropriate expert(s) per query
- Self-contained reasoning: Visible chain-of-thought

Usage:
    agent = MoEAgent()
    result = agent.process("Analyze NVDA using harmonic patterns")
    # Routes to: trading + consciousness experts
"""

import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ExpertConfig:
    """Configuration for a single expert adapter"""
    name: str
    adapter_path: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    description: str = ""
    trained: bool = False


@dataclass
class RouterDecision:
    """Router output - which experts to use"""
    experts: List[str]
    confidence: Dict[str, float]
    reasoning: str


class SimpleRouter:
    """
    Lightweight router that selects experts based on query analysis.

    Can be upgraded to:
    - Small classifier model (BERT, small Qwen)
    - Embedding-based similarity
    - LLM-based routing
    """

    def __init__(self, experts: Dict[str, ExpertConfig]):
        self.experts = experts
        self.keyword_map = self._build_keyword_map()

    def _build_keyword_map(self) -> Dict[str, List[str]]:
        """Map keywords to expert names"""
        keyword_map = defaultdict(list)
        for expert_name, config in self.experts.items():
            for keyword in config.keywords:
                keyword_map[keyword.lower()].append(expert_name)
        return keyword_map

    def route(self, query: str, context: Optional[Dict] = None) -> RouterDecision:
        """
        Determine which expert(s) should handle the query.

        Args:
            query: User query text
            context: Optional context (mode, history, etc.)

        Returns:
            RouterDecision with selected experts and reasoning
        """
        query_lower = query.lower()
        scores = defaultdict(float)

        # Keyword matching
        for keyword, expert_names in self.keyword_map.items():
            if keyword in query_lower:
                for expert_name in expert_names:
                    scores[expert_name] += 1.0

        # Context hints
        if context and 'mode' in context:
            mode = context['mode'].lower()
            if mode in self.experts:
                scores[mode] += 2.0

        # Domain-specific rules
        # Trading signals
        if any(word in query_lower for word in ['trade', 'stock', 'market', 'buy', 'sell', 'ticker']):
            scores['trading'] += 1.5

        # Code signals
        if any(word in query_lower for word in ['code', 'function', 'class', 'debug', 'implement']):
            scores['coding'] += 1.5

        # Research signals
        if any(word in query_lower for word in ['harmonic', 'consciousness', 'eigenmode', 'eigenvalue']):
            scores['consciousness'] += 1.5

        # Game signals
        if any(word in query_lower for word in ['game', 'unity', 'unreal', 'render', 'physics']):
            scores['gamedev'] += 1.5

        # Agent signals
        if any(word in query_lower for word in ['agent', 'coordinate', 'orchestrate', 'multi-step']):
            scores['orchestration'] += 1.5

        # Select top experts (threshold: 0.5)
        selected_experts = [name for name, score in scores.items() if score >= 0.5]

        # Default to general if no match
        if not selected_experts:
            selected_experts = ['general']
            scores['general'] = 0.3

        # Normalize scores to confidences
        total_score = sum(scores[e] for e in selected_experts)
        confidence = {e: scores[e] / total_score for e in selected_experts}

        # Generate reasoning
        reasoning = self._generate_reasoning(query, selected_experts, scores)

        return RouterDecision(
            experts=selected_experts,
            confidence=confidence,
            reasoning=reasoning
        )

    def _generate_reasoning(self, query: str, experts: List[str], scores: Dict[str, float]) -> str:
        """Generate human-readable routing reasoning"""
        lines = [
            f"Query: '{query[:50]}...' " if len(query) > 50 else f"Query: '{query}'",
            f"Matched {len(experts)} expert(s):"
        ]

        for expert in experts:
            score = scores[expert]
            lines.append(f"  - {expert}: {score:.2f} (confidence: {score/sum(scores.values()):.0%})")

        return "\n".join(lines)


class ReasoningChain:
    """Tracks internal reasoning steps for transparency"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.steps = []

    def add_step(self, step_name: str, content: str):
        """Add reasoning step"""
        self.steps.append({
            'step': step_name,
            'content': content,
            'timestamp': time.time()
        })

        if self.verbose:
            print(f"[AI Reasoning - {step_name}] {content}")

    def get_chain(self) -> str:
        """Get full reasoning chain as string"""
        return "\n".join([
            f"{s['step']}: {s['content']}"
            for s in self.steps
        ])

    def clear(self):
        """Clear reasoning history"""
        self.steps = []


class MoEAgent:
    """
    Multi-Expert Agent with LoRA adapter routing.

    Features:
    - Loads base model once (memory efficient)
    - Dynamically switches LoRA adapters
    - Routes queries to appropriate expert(s)
    - Internal reasoning chain (transparent)
    - Can combine multi-expert responses
    """

    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-32B-Instruct",
                 adapters_dir: str = "./adapters",
                 verbose: bool = True):
        """
        Initialize MoE Agent.

        Args:
            model_name: Base model to load
            adapters_dir: Directory containing LoRA adapters
            verbose: Show internal reasoning
        """
        self.model_name = model_name
        self.adapters_dir = Path(adapters_dir)
        self.verbose = verbose

        # Will be initialized lazily
        self.base_model = None
        self.tokenizer = None
        self.current_adapter = None

        # Expert configuration
        self.experts = self._initialize_experts()

        # Router and reasoning
        self.router = SimpleRouter(self.experts)
        self.reasoning = ReasoningChain(verbose=verbose)

        # Memory/context
        self.conversation_history = []

        print(f"[MoE Agent] Initialized with {len(self.experts)} experts")
        for name, config in self.experts.items():
            status = "[OK] Trained" if config.trained else "[X] Not trained"
            print(f"  - {name:15s}: {status} | {config.description}")

    def _initialize_experts(self) -> Dict[str, ExpertConfig]:
        """Define available experts and their configurations"""
        experts = {
            'general': ExpertConfig(
                name='general',
                adapter_path=self.adapters_dir / 'general_100k',  # Current training!
                keywords=['general', 'help', 'question'],
                description='General instruction following (100K OpenHermes)',
                trained=True  # Currently training
            ),
            'trading': ExpertConfig(
                name='trading',
                adapter_path=self.adapters_dir / 'trading_50k',
                keywords=['trade', 'stock', 'market', 'price', 'buy', 'sell', 'ticker',
                         'strategy', 'risk', 'portfolio', 'chart'],
                capabilities=['market_analysis', 'strategy_design', 'risk_assessment'],
                description='Trading and market analysis specialist',
                trained=False
            ),
            'coding': ExpertConfig(
                name='coding',
                adapter_path=self.adapters_dir / 'coding_100k',
                keywords=['code', 'function', 'class', 'debug', 'implement', 'algorithm',
                         'python', 'javascript', 'bug', 'optimize'],
                capabilities=['debug', 'implement', 'optimize', 'architect'],
                description='Software development and debugging',
                trained=False
            ),
            'consciousness': ExpertConfig(
                name='consciousness',
                adapter_path=self.adapters_dir / 'consciousness_10k',
                keywords=['consciousness', 'harmonic', 'eigenmode', 'eigenvalue', 'IIT',
                         'integration', 'phi', 'neural', 'brain'],
                capabilities=['theory_reasoning', 'experiment_analysis', 'harmonic_analysis'],
                description='Harmonic field consciousness research',
                trained=False
            ),
            'gamedev': ExpertConfig(
                name='gamedev',
                adapter_path=self.adapters_dir / 'gamedev_50k',
                keywords=['game', 'unity', 'unreal', 'render', 'shader', 'physics',
                         'engine', '3d', 'graphics', 'animation'],
                capabilities=['game_logic', 'physics_simulation', 'rendering', 'optimization'],
                description='Game development and engine work',
                trained=False
            ),
            'orchestration': ExpertConfig(
                name='orchestration',
                adapter_path=self.adapters_dir / 'orchestration_30k',
                keywords=['agent', 'coordinate', 'orchestrate', 'multi-step', 'pipeline',
                         'workflow', 'automation'],
                capabilities=['task_decomposition', 'coordination', 'multi_agent'],
                description='Agent orchestration and coordination',
                trained=False
            ),
        }

        # Check which adapters actually exist
        for expert in experts.values():
            if expert.adapter_path and expert.adapter_path.exists():
                expert.trained = True

        return experts

    def _load_model(self):
        """Lazy load base model (only when needed)"""
        if self.base_model is not None:
            return

        print(f"[MoE Agent] Loading base model: {self.model_name}...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            print(f"[MoE Agent] Model loaded successfully")

        except Exception as e:
            print(f"[MoE Agent] Error loading model: {e}")
            print(f"[MoE Agent] Falling back to mock mode for testing")
            self.base_model = "MOCK"  # Mock for testing

    def _load_adapter(self, expert_name: str):
        """Load LoRA adapter for specific expert"""
        if self.current_adapter == expert_name:
            return  # Already loaded

        expert = self.experts.get(expert_name)
        if not expert:
            raise ValueError(f"Unknown expert: {expert_name}")

        if not expert.trained:
            print(f"[WARNING] Expert '{expert_name}' not yet trained - using base model")
            self.current_adapter = None
            return

        # Load adapter using PEFT
        try:
            from peft import PeftModel

            if expert.adapter_path and expert.adapter_path.exists():
                print(f"[MoE Agent] Loading adapter: {expert_name}")

                # TODO: Implement actual adapter loading
                # For now, just track which adapter is "active"
                self.current_adapter = expert_name

            else:
                print(f"[WARNING] Adapter not found: {expert.adapter_path}")
                self.current_adapter = None

        except ImportError:
            print("[WARNING] PEFT not installed - using base model only")
            self.current_adapter = None

    def process(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main processing pipeline with expert routing.

        Args:
            query: User query
            context: Optional context dict (mode, data, etc.)

        Returns:
            Dict with response, reasoning, experts_used, confidence
        """
        self.reasoning.clear()
        self.reasoning.add_step("Input", f"Query: {query}")

        # Step 1: Route to expert(s)
        decision = self.router.route(query, context)
        self.reasoning.add_step("Routing", decision.reasoning)

        # Step 2: Process with each expert
        responses = []
        for expert_name in decision.experts:
            self.reasoning.add_step(f"Expert: {expert_name}",
                                   f"Processing with confidence {decision.confidence[expert_name]:.0%}")

            # Load appropriate adapter
            self._load_adapter(expert_name)

            # Generate response
            response = self._generate_response(query, context)
            responses.append((expert_name, response, decision.confidence[expert_name]))

        # Step 3: Synthesize if multiple experts
        if len(responses) > 1:
            final_response = self._synthesize_responses(query, responses)
            self.reasoning.add_step("Synthesis",
                                   f"Combined insights from {len(responses)} experts")
        else:
            final_response = responses[0][1]
            self.reasoning.add_step("Output", "Single expert response")

        return {
            'response': final_response,
            'experts_used': decision.experts,
            'confidence': decision.confidence,
            'reasoning_chain': self.reasoning.get_chain(),
            'timestamp': time.time()
        }

    def _generate_response(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Generate response using current adapter.

        This is where actual model inference happens.
        """
        # For now, return mock response
        # TODO: Implement actual inference with loaded adapter

        if self.base_model == "MOCK":
            return f"[MOCK RESPONSE] Processing '{query}' with expert '{self.current_adapter}'"

        # Actual implementation would be:
        # inputs = self.tokenizer(query, return_tensors="pt")
        # outputs = self.base_model.generate(**inputs)
        # response = self.tokenizer.decode(outputs[0])
        # return response

        return f"Response from {self.current_adapter} expert"

    def _synthesize_responses(self, query: str,
                              responses: List[Tuple[str, str, float]]) -> str:
        """
        Combine multiple expert responses into coherent answer.

        Args:
            query: Original query
            responses: List of (expert_name, response, confidence)

        Returns:
            Synthesized response
        """
        # Simple synthesis for now
        synthesis = f"[Multi-Expert Response]\n\n"
        synthesis += f"Query: {query}\n\n"

        for expert_name, response, confidence in responses:
            synthesis += f"### {expert_name.title()} Expert (Confidence: {confidence:.0%})\n"
            synthesis += f"{response}\n\n"

        # TODO: Use model to generate proper synthesis
        synthesis += "### Combined Recommendation\n"
        synthesis += "[Synthesis combining above expert insights would go here]\n"

        return synthesis


def demo():
    """Demo the MoE agent framework"""
    print("="*80)
    print("MOE AGENT FRAMEWORK DEMO")
    print("="*80)

    # Initialize agent
    agent = MoEAgent(verbose=True)

    # Test queries
    test_queries = [
        "Should I buy NVDA stock right now?",
        "Debug this Python function that's throwing an error",
        "Explain harmonic field theory of consciousness",
        "Help me optimize rendering in my Unity game",
        "Analyze AAPL using harmonic patterns in the price data",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {query}")
        print(f"{'='*80}\n")

        result = agent.process(query)

        print(f"\n[RESPONSE]")
        print(result['response'])

        print(f"\n[EXPERTS USED]: {', '.join(result['experts_used'])}")
        print(f"[CONFIDENCE]: {result['confidence']}")

        time.sleep(1)

    print(f"\n{'='*80}")
    print("DEMO COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    demo()
