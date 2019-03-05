
# def regularization_l2(l2_lambda: float) -> float:
#     all_w_sum = 0
#     for layer_idx in range(1, self.layers_count + 1):
#         all_w_sum += np.sum(self.params_values[f'w{layer_idx}'] ** 2)
#     return l2_lambda / 2 * all_w_sum
