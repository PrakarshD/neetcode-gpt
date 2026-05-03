class Solution:

    def function_x(self, input_value: float):
        return input_value * input_value
    
    def function_x_grad(self, input_value: float):
        return 2 * input_value

    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        optima = init

        for idx in range(iterations):
            derivative = self.function_x_grad(optima)

            optima = optima - learning_rate * derivative

        return round(optima,5)