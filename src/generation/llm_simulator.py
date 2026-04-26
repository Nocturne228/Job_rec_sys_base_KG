"""
LLM simulator for generating career advice.
Simulates Qwen-2.5 or similar LLM responses.
"""
import json
import random
from typing import Dict, List, Optional, Any
from datetime import datetime


class LLMSimulator:
    """Simulate LLM responses for career advice generation."""

    def __init__(self,
                 model_name: str = "qwen-2.5-simulated",
                 temperature: float = 0.3,
                 max_tokens: int = 1000,
                 seed: Optional[int] = None):
        """
        Initialize LLM simulator.

        Args:
            model_name: Simulated model name
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.temperature = max(0.0, min(1.0, temperature))
        self.max_tokens = max_tokens

        if seed is not None:
            random.seed(seed)

        # Predefined templates for different scenarios
        self.templates = self._load_templates()

        # Skill resources mapping
        self.skill_resources = {
            "python": ["Python.org tutorial", "Codecademy Python course", "Real Python articles"],
            "java": ["Oracle Java tutorials", "CodeGym", "Java Programming Masterclass on Udemy"],
            "javascript": ["MDN Web Docs", "JavaScript.info", "Eloquent JavaScript book"],
            "react": ["React documentation", "React Tutorial on egghead.io", "Fullstack Open course"],
            "vue": ["Vue.js documentation", "Vue Mastery", "Vue School"],
            "nodejs": ["Node.js documentation", "The Node.js Beginner Book", "Node.js Design Patterns"],
            "docker": ["Docker docs", "Docker Mastery course", "Docker Deep Dive"],
            "kubernetes": ["Kubernetes docs", "Kubernetes the Hard Way", "KubeAcademy"],
            "aws": ["AWS Training and Certification", "AWS Whitepapers", "A Cloud Guru"],
            "sql": ["SQLZoo", "Mode Analytics SQL tutorial", "SQL Bolt"],
            "pytorch": ["PyTorch tutorials", "Deep Learning with PyTorch", "Fast.ai course"],
            "tensorflow": ["TensorFlow tutorials", "TensorFlow Developer Certificate", "Coursera ML courses"]
        }

    def _load_templates(self) -> Dict[str, str]:
        """Load response templates."""
        return {
            "beginner": "As a beginner in {skill}, start with {resource1} to build fundamentals.",
            "intermediate": "To advance from intermediate to advanced in {skill}, focus on {resource1} and practice with {resource2}.",
            "advanced": "For expert-level mastery of {skill}, explore {resource1} and contribute to open-source projects.",
            "missing": "Since you don't have {skill}, begin with introductory materials like {resource1} before moving to {resource2}."
        }

    def generate(self,
                prompt: str,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate simulated LLM response.

        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Dictionary with simulated response
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Parse prompt to extract information
        context = self._parse_prompt(prompt)

        # Generate response based on context
        response = self._generate_response(context, temp)

        # Simulate token usage
        token_count = len(response.split())

        return {
            "model": self.model_name,
            "response": response,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": token_count,
                "total_tokens": len(prompt.split()) + token_count
            },
            "temperature": temp,
            "max_tokens": tokens,
            "timestamp": datetime.now().isoformat()
        }

    def _parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse prompt to extract key information."""
        # Simplified parsing - in reality would use more sophisticated NLP
        context = {
            "skills": [],
            "skill_gaps": [],
            "coverage": 0.0,
            "has_json_request": "JSON" in prompt
        }

        # Extract skill mentions
        common_skills = list(self.skill_resources.keys())
        for skill in common_skills:
            if skill.lower() in prompt.lower():
                context["skills"].append(skill)

        # Try to extract coverage percentage
        import re
        coverage_match = re.search(r'Skill Coverage:\s*([\d.]+)%', prompt)
        if coverage_match:
            context["coverage"] = float(coverage_match.group(1)) / 100

        # Extract skill gaps from prompt
        gap_section = None
        lines = prompt.split('\n')
        in_gap_section = False

        for line in lines:
            if "Skill Gap Analysis" in line:
                in_gap_section = True
                continue
            if in_gap_section and line.strip() and ":" in line:
                # Parse gap line like "- python: Current=None, Required=intermediate"
                parts = line.strip().lstrip('- ').split(':')
                if len(parts) >= 2:
                    skill_id = parts[0].strip()
                    details = parts[1].strip()
                    context["skill_gaps"].append({
                        "skill": skill_id,
                        "details": details
                    })

        return context

    def _generate_response(self, context: Dict[str, Any], temperature: float) -> str:
        """Generate response based on context."""
        # Adjust randomness based on temperature
        randomness = temperature

        # Determine response type
        if context["has_json_request"]:
            # Generate structured JSON response
            return self._generate_json_response(context, randomness)
        else:
            # Generate free-text response
            return self._generate_text_response(context, randomness)

    def _generate_json_response(self, context: Dict[str, Any], randomness: float) -> str:
        """Generate JSON-formatted response."""
        skill_gaps = context.get("skill_gaps", [])
        coverage = context.get("coverage", 0.5)

        # Determine critical skills (first 3 gaps or random selection)
        if skill_gaps:
            critical_skills = [gap["skill"] for gap in skill_gaps[:3]]
        else:
            critical_skills = context.get("skills", ["python", "sql", "communication"])[:3]

        # Generate learning path
        learning_path = []
        for i, skill in enumerate(critical_skills[:5]):  # Max 5 skills in path
            # Determine target level
            if skill_gaps and i < len(skill_gaps):
                details = skill_gaps[i]["details"]
                if "Current=None" in details:
                    current_level = "beginner"
                    target_level = "intermediate"
                else:
                    current_level = "intermediate"
                    target_level = "advanced"
            else:
                current_level = random.choice(["beginner", "intermediate"])
                target_level = "advanced" if current_level == "intermediate" else "intermediate"

            # Get resources
            resources = self.skill_resources.get(skill.lower(), [
                f"Online {skill} course",
                f"{skill} documentation",
                f"Practice {skill} projects"
            ])

            # Add some randomness
            if randomness > 0.5 and len(resources) > 1:
                resources = random.sample(resources, min(2, len(resources)))

            learning_path.append({
                "skill_id": skill,
                "current_level": current_level,
                "target_level": target_level,
                "resources": resources[:2],  # Max 2 resources
                "estimated_time": f"{random.randint(1, 4)}-{random.randint(3, 8)} months",
                "priority": "high" if i == 0 else ("medium" if i == 1 else "low")
            })

        # Calculate timeline
        timeline_months = max(3, min(12, int(10 * (1 - coverage) + random.randint(0, 6))))

        # Confidence score based on coverage and skill count
        confidence = min(0.9, 0.3 + coverage * 0.5 + len(context["skills"]) * 0.05)
        confidence += random.uniform(-0.1, 0.1) * randomness  # Add some noise

        response_data = {
            "summary": f"Based on your {coverage:.0%} skill coverage for the target job, focus on mastering {', '.join(critical_skills[:2])} first.",
            "critical_skills": critical_skills,
            "learning_path": learning_path,
            "timeline_months": timeline_months,
            "confidence_score": round(confidence, 2)
        }

        return json.dumps(response_data, indent=2)

    def _generate_text_response(self, context: Dict[str, Any], randomness: float) -> str:
        """Generate free-text response."""
        skill_count = len(context.get("skills", []))
        coverage = context.get("coverage", 0.5)

        # Base response
        if coverage > 0.8:
            base = "You have strong alignment with the job requirements."
            advice = "Focus on refining your existing skills and preparing for interviews."
        elif coverage > 0.5:
            base = "You have moderate alignment with the job requirements."
            advice = "You need to build a few key skills to be competitive."
        else:
            base = "There's significant skill gap between your current profile and the job requirements."
            advice = "Consider a structured learning path over several months."

        # Add skill-specific advice
        skill_advice = []
        for skill in context.get("skills", [])[:3]:
            resources = self.skill_resources.get(skill.lower(), [f"Learn {skill}"])
            skill_advice.append(f"For {skill}, check out {resources[0]}.")

        # Combine
        response = f"{base} {advice}\n\n"
        if skill_advice:
            response += "Specific recommendations:\n" + "\n".join(skill_advice)

        # Add general advice
        response += "\n\nGeneral advice: Focus on practical projects, network with professionals in the field, "
        response += "and consider getting relevant certifications if available."

        return response

    def batch_generate(self,
                      prompts: List[str],
                      temperature: Optional[float] = None) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts."""
        return [self.generate(prompt, temperature) for prompt in prompts]

    def evaluate_response_quality(self, response: str) -> Dict[str, float]:
        """
        Evaluate the quality of a generated response.
        Simulates LLM-as-a-Judge evaluation.
        """
        # Simple heuristics for evaluation
        score = 0.5  # Base score

        # Check for JSON structure
        try:
            data = json.loads(response)
            score += 0.2  # Well-formatted JSON
            if "learning_path" in data and data["learning_path"]:
                score += 0.1
            if "confidence_score" in data:
                score += 0.05
        except:
            # Not JSON, check for structure
            if len(response) > 100:
                score += 0.1
            if "recommend" in response.lower() or "suggest" in response.lower():
                score += 0.05

        # Length bonus (but not too long)
        token_count = len(response.split())
        if 50 < token_count < 500:
            score += 0.1
        elif token_count >= 500:
            score += 0.05

        # Cap score
        score = min(0.95, max(0.1, score))

        return {
            "overall": score,
            "formatting": 0.8 if "{" in response and "}" in response else 0.3,
            "completeness": min(0.9, token_count / 200),
            "specificity": 0.6 if any(skill in response.lower() for skill in self.skill_resources) else 0.3
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the simulated model."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "simulated": True,
            "templates_available": len(self.templates),
            "skills_covered": len(self.skill_resources)
        }