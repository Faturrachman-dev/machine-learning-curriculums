"""
DataCalculator: A class for performing statistical calculations on numerical data.
This implementation serves as a foundation for understanding basic statistical operations
commonly used in data science and machine learning.

Author: [Your Name]
Date: [Current Date]
"""

from typing import List, Union
import math

class DataCalculator:
    """
    A calculator class for performing statistical operations on numerical data.
    
    Attributes:
        data (List[float]): List storing numerical values
        _last_operation (str): Tracks the last operation performed (for logging)
    """
    
    def __init__(self):
        """Initialize an empty DataCalculator instance."""
        self.data = []
        self._last_operation = None
    
    def add_number(self, num: Union[int, float]) -> None:
        """
        Add a number to the dataset.
        
        Args:
            num (Union[int, float]): Number to add to the dataset
            
        Raises:
            TypeError: If num is not a number
        """
        if not isinstance(num, (int, float)):
            raise TypeError("Input must be a number")
        self.data.append(float(num))
        self._last_operation = f"Added number: {num}"
    
    def calculate_mean(self) -> float:
        """
        Calculate the arithmetic mean of the dataset.
        
        Returns:
            float: Mean value, or 0 if dataset is empty
            
        Example:
            >>> calc = DataCalculator()
            >>> calc.add_number(1)
            >>> calc.add_number(2)
            >>> calc.calculate_mean()
            1.5
        """
        if not self.data:
            return 0
        result = sum(self.data) / len(self.data)
        self._last_operation = f"Calculated mean: {result}"
        return result
    
    def calculate_median(self) -> float:
        """
        Calculate the median value of the dataset.
        
        Returns:
            float: Median value, or 0 if dataset is empty
        """
        if not self.data:
            return 0
        sorted_data = sorted(self.data)
        n = len(sorted_data)
        mid = n // 2
        
        if n % 2 == 0:
            result = (sorted_data[mid-1] + sorted_data[mid]) / 2
        else:
            result = sorted_data[mid]
            
        self._last_operation = f"Calculated median: {result}"
        return result
    
    def calculate_variance(self) -> float:
        """
        Calculate the variance of the dataset.
        
        Returns:
            float: Variance value, or 0 if dataset has less than 2 values
        """
        if len(self.data) < 2:
            return 0
        
        mean = self.calculate_mean()
        squared_diff_sum = sum((x - mean) ** 2 for x in self.data)
        result = squared_diff_sum / (len(self.data) - 1)  # Using sample variance
        
        self._last_operation = f"Calculated variance: {result}"
        return result
    
    def calculate_std_dev(self) -> float:
        """
        Calculate the standard deviation of the dataset.
        
        Returns:
            float: Standard deviation value, or 0 if dataset has less than 2 values
        """
        result = math.sqrt(self.calculate_variance())
        self._last_operation = f"Calculated standard deviation: {result}"
        return result
    
    def get_summary(self) -> dict:
        """
        Get a summary of basic statistical measures.
        
        Returns:
            dict: Dictionary containing mean, median, variance, and standard deviation
        """
        summary = {
            'count': len(self.data),
            'mean': self.calculate_mean(),
            'median': self.calculate_median(),
            'variance': self.calculate_variance(),
            'std_dev': self.calculate_std_dev(),
            'min': min(self.data) if self.data else 0,
            'max': max(self.data) if self.data else 0
        }
        self._last_operation = "Generated summary statistics"
        return summary
    
    def get_last_operation(self) -> str:
        """
        Get the description of the last operation performed.
        
        Returns:
            str: Description of the last operation, or None if no operation performed
        """
        return self._last_operation

def main():
    """Example usage of the DataCalculator class."""
    # Create a calculator instance
    calc = DataCalculator()
    
    # Add some sample data
    sample_data = [2.5, 3.0, 4.5, 5.0, 6.0]
    for num in sample_data:
        calc.add_number(num)
        print(f"Added number: {num}")
    
    # Get and print summary statistics
    summary = calc.get_summary()
    print("\nSummary Statistics:")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()