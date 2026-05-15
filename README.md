# Netflix Multi-Objective Recommendation System

Thompson Sampling contextual bandit algorithm balancing engagement, discovery, and diversity.

## Overview

A machine learning system that demonstrates how Netflix personalizes content recommendations using **Thompson Sampling** — a Bayesian bandit algorithm that learns the optimal balance between three competing objectives:

- **Engagement (60%)**: Users watch content they'll likely enjoy
- **Discovery (30%)**: Users find new titles they wouldn't discover themselves  
- **Diversity (10%)**: Recommendations span different genres and creators

## The Problem

Recommendation systems face a fundamental tradeoff:

| Strategy | Result | Problem |
|----------|--------|---------|
| **Exploit Only** (100% familiar) | High engagement | User boredom → churn |
| **Explore Only** (100% new) | High discovery | User frustration → churn |
| **Balanced** (90% familiar, 10% new) | Both engagement AND discovery | Optimal ✓ |

Netflix's challenge: Find the right balance **for each user**.

## The Solution: Thompson Sampling

Thompson Sampling is a Bayesian bandit algorithm that:

1. **Maintains Bayesian posteriors** over recommendation strategy performance
2. **Samples intelligently** from those posteriors (not random)
3. **Biases toward winners** while occasionally exploring alternatives
4. **Learns per-user** strategies via contextual bandits

### Why Thompson Sampling?

- ✓ Intelligent exploration (Bayesian vs random)
- ✓ Converges 2x faster than epsilon-greedy alternatives
- ✓ Natural uncertainty quantification
- ✓ A/B test efficient (detects effects faster)

## How It Works

### Five Recommendation Strategies

The system tests 5 different explore/exploit ratios:

```
Strategy 0: Full Exploit          (0% new, 100% familiar)
Strategy 1: 90% Exploit/10% Explore  (90% familiar, 10% new)
Strategy 2: 50-50 Mix             (50% familiar, 50% new)
Strategy 3: 10% Exploit/90% Explore  (10% familiar, 90% new)
Strategy 4: Full Explore          (100% new, 0% familiar)
```

### Thompson Sampling Process

```
1. Maintain Beta(alpha, beta) posterior for each strategy
   alpha = successes (user watched)
   beta = failures (user skipped)

2. Sample at decision time:
   sampled_reward[i] = Beta(alpha[i], beta[i]).sample()

3. Pick strategy with highest sample:
   strategy = argmax(sampled_rewards)

4. Get recommendation based on selected strategy

5. Observe feedback (user watched or skipped)

6. Update posterior:
   if watched: alpha[strategy] += 1
   else: beta[strategy] += 1
```

### Multi-Objective Reward Function

```python
reward = 0.6 * engagement + 0.3 * discovery + 0.1 * diversity

engagement (0-1):  1.0 if user watched, 0.0 if skipped
discovery (0-1):   1.0 if new to user, 0.5 if familiar
diversity (0-1):   1.0 if new genre, 0.3 if familiar genre
```

The weights (0.6/0.3/0.1) explicitly encode business priorities:
- Engagement matters most (revenue signal)
- Discovery matters for retention (long-term)
- Diversity is secondary (quality of life)

## Example Output

```
Running 15 recommendation rounds...
────────────────────────────────────────────────────────────────
Rd | Strategy                            | Title        | Watched | Reward
────────────────────────────────────────────────────────────────
 1 | 90% Exploit / 10% Explore           | drama_1      | ✓       | 0.780
 2 | 90% Exploit / 10% Explore           | action_1     | ✓       | 0.780
 3 | 90% Exploit / 10% Explore           | drama_1      | ✓       | 0.780
...
────────────────────────────────────────────────────────────────

Thompson Sampling Results:
────────────────────────────────────────────────────────────────
Strategy                            | Successes  | Rate    | Uncertainty
────────────────────────────────────────────────────────────────
90% Exploit / 10% Explore           |        14 |  93.3% |    0.003889
Full Exploit                        |         5 |  71.4% |    0.025510
50-50 Mix                           |         2 |  50.0% |    0.050000
10% Exploit / 90% Explore           |         1 |  50.0% |    0.083333
Full Explore                        |         1 |  50.0% |    0.083333

SUCCESS! Thompson Sampling learned which strategies work best.
```

**Key insight**: Thompson Sampling selected "90% Exploit / 10% Explore" **14 out of 15 times** because it learned this strategy had the highest success rate (93.3%). The algorithm automatically learned the best strategy without manual tuning.

## Installation

### Requirements

- Python 3.6+
- NumPy (only dependency!)

```bash
pip install numpy
```

### Quick Start

```bash
python netflix_recommendation_code.py
```

The script will:
1. Initialize Thompson Sampling
2. Run 15 recommendation rounds
3. Show which strategies Thompson Sampling selects
4. Display final results showing which strategies "won"

## Understanding the Results

### Success Rate
Percentage of times users watched recommendations from each strategy.
- Higher = better strategy
- Thompson Sampling biases toward higher rates

### Uncertainty
Confidence in the success rate estimate.
- Lower uncertainty = more confident in the estimate
- Higher uncertainty = less data about this strategy (explored less)

**Example**: "90% Exploit / 10% Explore" has uncertainty 0.003889 (very low) and 93.3% success rate → Thompson Sampling is **very confident** this is the best strategy → it uses it frequently.

## Key Concepts Explained

### Explore vs Exploit

**Exploit**: Recommend content similar to what user already watches (high engagement risk, low discovery potential)

**Explore**: Recommend new content (discovery potential, high engagement risk)

**Thompson Sampling learns**: The right balance for each user.

### Bayesian Posterior

A probability distribution representing our belief about strategy performance.

**Example**: After seeing "90% Exploit / 10% Explore" succeed 14 times and fail 1 time:
- Posterior: Beta(14, 1)
- Belief: This strategy succeeds ~93% of the time
- Confidence: Very high (low uncertainty)

### Contextual Bandit

A bandit problem where decisions depend on context (user information).

**Unlike standard bandit**: Not one strategy for everyone
**Like real Netflix**: Each user gets a personalized strategy

## Technologies

- **Python**: Implementation language
- **NumPy**: Fast numerical computing (Beta distributions, sampling)
- **Bayesian Inference**: Thompson Sampling posteriors
- **Multi-Objective Optimization**: Reward function with explicit weights
- **A/B Testing**: Validation methodology

## What This Demonstrates

✓ **Multi-objective machine learning** - Balancing competing goals with explicit weights  
✓ **Bayesian inference** - Using posterior distributions for decision-making  
✓ **Bandit algorithms** - Exploration/exploitation tradeoff  
✓ **Production ML thinking** - Validation, monitoring, real-world constraints  
✓ **Clear communication** - Explaining complex ML concepts  

## Interview Talking Points

**When asked "Tell us about your project":**

> "I built a Thompson Sampling recommendation system that demonstrates how Netflix personalizes content. The challenge: balance engagement (watch time) with discovery (new content). I modeled this as a contextual bandit—each 'arm' is a strategy (0-100% new content), and the context is user watch history. Thompson Sampling maintains Bayesian posteriors over strategy success rates. It samples from these posteriors to decide which strategy to use—smart exploration, not random. My code shows this working: after 15 rounds, Thompson Sampling learned that '90% Exploit / 10% Explore' was best (93.3% success), so it used that strategy 14 out of 15 times. The reward function balances three objectives: 60% engagement + 30% discovery + 10% diversity. This ensures the algorithm learns a sustainable balance, not just maximum engagement."

**Key phrases Netflix wants to hear:**
- Thompson Sampling / Bayesian bandits
- Multi-objective optimization
- Exploration/exploitation tradeoff
- Intelligent learning (not random)
- Production ML thinking

## Files

- `netflix_recommendation_code.py` - Main implementation (pure NumPy)
- `EXPLORE_VS_EXPLOIT_EXPLAINED.md` - Detailed concept explanation
- `NETFLIX_INTERVIEW_GUIDE.md` - Interview preparation guide
- `README.md` - This file

## Next Steps

1. **Run the code**: `python netflix_recommendation_code.py`
2. **Understand the output**: Which strategies does Thompson Sampling select?
3. **Read the concepts**: See EXPLORE_VS_EXPLOIT_EXPLAINED.md
4. **Interview prep**: See NETFLIX_INTERVIEW_GUIDE.md

## Key Insights

1. **Thompson Sampling learns intelligently** - Biases toward high-performing strategies
2. **Bayesian approach works** - Uncertainty quantification guides learning
3. **Context matters** - Different users need different strategies
4. **Multi-objective is hard** - Explicit weights are essential
5. **Online validation matters** - Offline results may differ from real-world

## Extensions

**Possible improvements:**
- Expand from 5 to 10+ strategies
- User segmentation (different weights for different user types)
- Temporal models (preferences drift over time)
- Causal inference (does discovery actually cause retention?)

## Author

Saroj Bono  
ML Engineer | 4 years ML development + 21 years software engineering  
850-704-2599 | bonoa201@comcast.net

## License

MIT

---

**Made for Netflix ML Engineer interviews. Demonstrates complete understanding of multi-objective ML, Thompson Sampling, Bayesian inference, and production ML thinking.**