# ISL

ISL: Monitor for Image Segmentation Logic

## Overview

ISL provides a logic-based framework for specifying and monitoring properties of image segmentation tasks. It uses a custom logic language and parser to describe and check properties over segmented images.

## Features

- Custom logic grammar for expressing segmentation properties
- Parsing and evaluation using [parsimonious](https://github.com/erikrose/parsimonious)
- Extensible visitor pattern for logic evaluation

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/isl.git
   cd isl
   ```

2. Install the required Python package:
   ```sh
   pip install -r requirements.txt
   ```

## Usage 

You can use the logic parser in your Python code:

```sh
from parsimonious import Grammar
from dscl import DSCLLogicVisitor, grammar_text

grammar = Grammar(grammar_text)
tree = grammar.parse('your_formula_here')
visitor = DSCLLogicVisitor()
result = visitor.visit(tree)
```

Replace 'your_formula_here' with your logic formula.

## Requirements

- Python 3.7+
- parsimonious

## License

MIT License