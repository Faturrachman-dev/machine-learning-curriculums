# Module 1: Robust Python Programming for Data Science & ML

## Learning Path Overview

This module will help you rebuild your Python coding muscle memory through hands-on practice. We'll focus on writing code ourselves rather than relying on AI generation.

### Week 1: Python Fundamentals Review

#### Day 1-2: Basic Python Refresher
- **Mini-Project:** Build a Data Structure Calculator
```python
# Example structure of what we'll build:
class DataCalculator:
    def __init__(self):
        self.data = []
    
    def add_number(self, num):
        self.data.append(num)
    
    def calculate_mean(self):
        return sum(self.data) / len(self.data) if self.data else 0
```

#### Day 3-4: Control Flow & Functions
- **Challenge:** Create a Number Guessing Game
- Practice with:
  - if/else statements
  - while/for loops
  - function definitions
  - error handling

#### Day 5: List Comprehensions & Generators
- **Exercise:** Data Processing Pipeline
- Compare performance between:
  - Traditional loops
  - List comprehensions
  - Generator expressions

### Week 2: Advanced Python Concepts

#### Day 6-7: Object-Oriented Programming
- **Project:** Build a Dataset Handler Class
- Implement:
  - Class inheritance
  - Method overriding
  - Property decorators
  - Static methods

#### Day 8-9: Decorators & Context Managers
- **Exercise:** Create Custom Decorators for:
  - Timing function execution
  - Logging function calls
  - Error handling

#### Day 10: Exception Handling & Testing
- **Project:** Error-Proof Data Processor
- Practice:
  - Try/except blocks
  - Custom exceptions
  - Unit testing

## Daily Challenges

### Today's Challenge: Basic Calculator
Let's start with a simple but comprehensive calculator that uses various Python concepts:

1. Create a new file `calculator.py` in your module directory
2. Implement these features:
   - Basic arithmetic operations
   - Input validation
   - History tracking
   - Error handling

Here's your starter template:

```python
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, x, y):
        # TODO: Implement addition and history tracking
        pass
    
    def subtract(self, x, y):
        # TODO: Implement subtraction and history tracking
        pass
    
    def multiply(self, x, y):
        # TODO: Implement multiplication and history tracking
        pass
    
    def divide(self, x, y):
        # TODO: Implement division with error handling
        pass
    
    def get_history(self):
        # TODO: Return formatted history of operations
        pass
```

### Your Task:
1. Implement all the TODO sections
2. Add input validation
3. Add proper error handling
4. Add docstrings and comments
5. Create a main() function to test your calculator

### Bonus Challenges:
- Add memory functions (M+, M-, MR, MC)
- Add support for more operations (power, square root, etc.)
- Add unit tests
- Add a command-line interface

## Resources
- [Python Official Documentation](https://docs.python.org/3/)
- [Real Python Tutorials](https://realpython.com/)
- [Python Koans](https://github.com/gregmalcolm/python_koans)

## Next Steps
After completing the calculator project, we'll move on to:
1. Data structure implementations
2. Algorithm challenges
3. Python for data processing

Remember: The goal is to write the code yourself without AI assistance to rebuild your coding muscle memory! 