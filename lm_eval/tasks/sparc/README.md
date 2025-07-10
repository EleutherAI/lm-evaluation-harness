# SPaRC: Spatial Pathfinding Reasoning Challenge

## Paper

**SPaRC: A Spatial Pathfinding Reasoning Challenge**

Abstract: Existing reasoning datasets saturate and fail to test abstract, multi-step problems, especially pathfinding and complex rule constraint satisfaction. We introduce SPaRC (Spatial Pathfinding Reasoning Challenge), a dataset of 1,000 2D grid pathfinding puzzles to evaluate spatial and symbolic reasoning, requiring step-by-step planning with arithmetic and geometric rules. Humans achieve near-perfect accuracy (98.0%; 94.5% on hard puzzles), while the best reasoning models, such as o4-mini, struggle (15.8%; 1.1% on hard puzzles). Models often generate invalid paths (>50% of puzzles for o4-mini), and reasoning tokens reveal they make errors in navigation and spatial logic. Unlike humans, who take longer on hard puzzles, models fail to scale test-time compute with difficulty. Allowing models to make multiple solution attempts improves accuracy, suggesting potential for better spatial reasoning with improved training and efficient test-time scaling methods. SPaRC can be used as a window into models' spatial reasoning limitations and drive research toward new methods that excel in abstract, multi-step problem-solving.

**Paper**: [arXiv:2505.16686](https://arxiv.org/abs/2505.16686)

**Authors**: Lars Benedikt Kaesberg, Jan Philip Wahle, Terry Ruas, Bela Gipp

**Homepage**: [https://sparc.gipplab.org](https://sparc.gipplab.org)

**Dataset**: [https://huggingface.co/datasets/lkaesberg/SPaRC](https://huggingface.co/datasets/lkaesberg/SPaRC)

**Code**: [https://github.com/lkaesberg/SPaRC](https://github.com/lkaesberg/SPaRC)

## Citation

```bibtex
@misc{kaesberg2025sparc,
      title={SPaRC: A Spatial Pathfinding Reasoning Challenge},
      author={Lars Benedikt Kaesberg and Jan Philip Wahle and Terry Ruas and Bela Gipp},
      year={2025},
      eprint={2505.16686},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.16686}
}
```

## Task Description

SPaRC is a spatial pathfinding reasoning challenge designed to evaluate language models' ability to understand and solve 2D grid-based pathfinding puzzles inspired by "The Witness" game mechanics. The task requires models to:

1. **Parse spatial information**: Understand 2D grid layouts with various constraints and rule symbols
2. **Apply logical reasoning**: Navigate through complex rule systems including squares, stars, polyshapes, and triangles  
3. **Generate valid paths**: Produce sequences of coordinates that form valid solutions with proper formatting
4. **Handle multi-step planning**: Execute step-by-step reasoning for complex spatial problems
5. **Understand game mechanics**: Comprehend region-based rules, path constraints, and symbol interactions

## Dataset Structure

The SPaRC dataset contains 1,000 2D grid pathfinding puzzles with varying difficulty levels. Each puzzle includes:

- **Grid specifications**: Height, width, and layout information
- **Puzzle constraints**: Rules and obstacles defined in puzzle arrays
- **Visualization**: Text-based representations of the grid
- **Solutions**: Valid path sequences that solve the puzzle
- **Difficulty metrics**: Complexity scores and difficulty levels
- **Special shapes**: Polyshape definitions for complex constraints

## Key Features

- **Spatial reasoning evaluation**: Tests models' understanding of 2D spatial relationships
- **Multi-step problem solving**: Requires planning and sequential reasoning
- **Custom validation**: Implements path validity checking beyond simple string matching
- **Difficulty gradation**: Puzzles range from simple to highly complex
- **Robust evaluation**: Multiple metrics including exact match and path validity

## Metrics

The task uses comprehensive evaluation metrics for detailed spatial reasoning assessment:

### Primary Metrics
1. **Exact Match**: Direct comparison between predicted and ground truth paths
2. **Path Validity**: Overall validation against known solutions in the dataset

### Detailed Path Analysis Metrics
3. **Starts at Start, Ends at Exit**: Whether the path correctly begins at 'S' and ends at 'E'
4. **Connected Line**: Whether all path segments are properly connected (adjacent cells)
5. **Non-Intersecting Line**: Whether the path avoids crossing itself
6. **Start to Exit Connected**: Combined metric for proper start-to-end connectivity
7. **No Rule Crossing**: Whether the path avoids rule cells (cells where both x,y coordinates are odd)
8. **Fully Valid Path**: Comprehensive validation combining all spatial reasoning requirements

## Enhanced Processing

The implementation includes several advanced components:

- **Smart Path Extraction**: Robust parsing with support for "####" solution markers
- **Multi-level Validation**: From basic format checking to complete spatial reasoning validation
- **Rule Constraint Checking**: Validates adherence to spatial rule constraints
- **Detailed Analysis Pipeline**: Provides granular metrics for different aspects of spatial reasoning
- **Format Handling**: Support for multiple coordinate formats (e.g., (x,y), [x,y])
- **Error Recovery**: Graceful handling of malformed outputs

## Usage

```bash
# Run SPaRC evaluation
python -m lm_eval --model hf \
    --model_args pretrained=<model_name> \
    --tasks sparc \
    --batch_size 1
```

## Performance Expectations

Based on the original paper:
- **Human performance**: 98.0% accuracy (94.5% on hard puzzles)
- **Best models (o4-mini)**: 15.8% accuracy (1.1% on hard puzzles)
- **Common failure modes**: Invalid path generation (>50% of cases for top models)

### Expected Metric Breakdown
The detailed metrics help identify specific spatial reasoning failure modes:
- **Path Extraction**: Models often struggle to format coordinate sequences correctly
- **Start/Exit Points**: Basic navigation understanding varies significantly across models
- **Path Connectivity**: Many models generate disconnected or invalid movement sequences
- **Rule Adherence**: Advanced constraint satisfaction (avoiding rule cells) proves challenging
- **Overall Validity**: Combined spatial reasoning requirements show the full challenge scope

## Implementation Notes

- Uses comprehensive prompt format explaining The Witness puzzle mechanics with examples
- Implements generative evaluation with enhanced path extraction supporting "####" solution markers
- Implements multi-stage filter pipeline: extraction → validation → detailed analysis
- Provides granular spatial reasoning metrics beyond simple accuracy
- Validates rule constraints specific to SPaRC puzzles (odd-coordinate rule cells)
- Includes detailed symbol legend and rule explanations in prompts
- Supports various coordinate formats and representations
- Includes comprehensive error handling for malformed outputs
- Designed to work with both instruction-tuned and base language models
- Reports 8 different metrics for comprehensive spatial reasoning assessment

### Groups and Tasks

#### Groups
- `spatial_reasoning`
- `pathfinding`

#### Tasks
- `sparc`: Main SPaRC pathfinding challenge

### Checklist

For adding novel benchmarks/datasets to the library:
- [x] Is the task an existing benchmark in the literature?
  - [x] Have you referenced the original paper that introduced the task?
  - [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
- [x] Is the "Main" variant of this task clearly denoted?
- [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
- [x] Have you noted which, if any, published evaluation setups are matched by this variant? 