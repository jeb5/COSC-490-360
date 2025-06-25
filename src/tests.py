from angle_estimation_helpers import entire_transformation
import torch
from scipy.spatial.transform import Rotation as R


# --- Replace with your mystery function ---
def mystery_function(A):
  A = A.numpy()
  A = entire_transformation(A)
  A = torch.tensor(A, dtype=torch.float32)
  return A


# -------------------------------------------


# Generate torch rotation matrices
def generate_rotations():
  angles = [1, 30, 60, 91, 120]
  rots = []
  for yaw in angles:
    for pitch in angles:
      for roll in angles:
        mat = R.from_euler("zyx", [yaw, pitch, roll], degrees=True).as_matrix()
        rots.append(torch.tensor(mat, dtype=torch.float32))
  return rots


A_list = generate_rotations()
B_list = [mystery_function(A) for A in A_list]


# ==== TEST 1: Linear model f(A) = M @ A ====
def test_linear_model(A_list, B_list):
  X = torch.stack([A.reshape(-1) for A in A_list])  # (N, 9)
  Y = torch.stack([B.reshape(-1) for B in B_list])  # (N, 9)
  M_flat, _res, _rank, _sing = torch.linalg.lstsq(X, Y)
  Y_pred = X @ M_flat
  err = torch.norm(Y - Y_pred)
  return err.item()


# ==== TEST 2: Conjugation model f(A) = M @ A @ M⁻¹ ====
def test_conjugation(A_list, B_list):
  A1, B1 = A_list[0], B_list[0]

  # Solve A X = X B for X using least squares
  def sylvester_solve(A, B):
    # Solve AX = XB -> reshape into AX - XB = 0
    # AX - XB = 0 → (I ⊗ A - Bᵗ ⊗ I) vec(X) = 0
    I = torch.eye(3)
    K = torch.kron(I, A) - torch.kron(B.T, I)
    _, _, V = torch.linalg.svd(K)
    x = V[-1]  # smallest singular vector
    X = x.reshape(3, 3)
    return X

  X = sylvester_solve(A1, B1)
  if torch.linalg.matrix_rank(X) < 3:
    return float("inf")
  M = torch.linalg.inv(X)
  total_err = 0.0
  for A, B in zip(A_list, B_list):
    B_pred = M @ A @ torch.linalg.inv(M)
    total_err += torch.norm(B - B_pred)
  return (total_err / len(A_list)).item()


# # ==== TEST 3: Sandwich model f(A) = M1 @ A @ M2 ====
# def test_sandwich(A_list, B_list):
#   K_list = []
#   b_list = []
#   for A, B in zip(A_list, B_list):
#     K = torch.kron(torch.eye(3), A.T)  # shape (9, 9)
#     b = B.T.reshape(-1, 1)
#     K_list.append(K)
#     b_list.append(b)

#   K_total = torch.vstack(K_list)
#   b_total = torch.vstack(b_list)
#   x, _res, _rank, _sing = torch.linalg.lstsq(K_total, b_total)
#   total_err = 0.0
#   for A, B in zip(A_list, B_list):
#     K = torch.kron(torch.eye(3), A.T)
#     B_vec_pred = K @ x
#     B_vec_true = B.T.reshape(-1, 1)
#     total_err += torch.norm(B_vec_true - B_vec_pred)
#   return (total_err / len(A_list)).item()


# Run tests
err_linear = test_linear_model(A_list, B_list)
err_conj = test_conjugation(A_list, B_list)
# err_sandwich = test_sandwich(A_list, B_list)

print("Linear model error:      ", err_linear)
print("Conjugation model error: ", err_conj)
# print("Sandwich model error:    ", err_sandwich)


def get_flat_linear_map(A_list, B_list):
  X = torch.stack([A.reshape(-1) for A in A_list])  # shape (N, 9)
  Y = torch.stack([B.reshape(-1) for B in B_list])  # shape (N, 9)
  L, _res, _rank, _sing = torch.linalg.lstsq(X, Y)  # shape (9, 9)
  return L.T  # L maps vec(A) → vec(B)


L = get_flat_linear_map(A_list, B_list)

# Test it
err = 0
for A, B in zip(A_list, B_list):
  A_flat = A.reshape(-1)
  B_pred = (L @ A_flat).reshape(3, 3)
  err += torch.norm(B - B_pred)
print("Flat model average error:", err / len(A_list))

torch.set_printoptions(precision=1, sci_mode=False)
print("Flat linear map L:")
print(L)

M_entries = [f"M{i}{j}" for i in range(3) for j in range(3)]

# Convert to tensor of strings for easier indexing
M_vec = M_entries

# Apply transformation: each entry of M' is a linear combo of M entries with signs
M_prime_vec = []
for row in L:
  expr_parts = []
  for coeff, var in zip(row, M_vec):
    if abs(coeff) > 1e-6:  # ignore coefficients very close to zero
      sign = "-" if coeff < 0 else ""
      coeff_abs = abs(coeff)
      if abs(coeff_abs - 1) < 1e-6:
        expr_parts.append(f"{sign}{var}")
      else:
        expr_parts.append(f"{sign}{coeff_abs:.3f}*{var}")
  if expr_parts:
    expr = " + ".join(expr_parts).replace("+ -", "- ")
  else:
    expr = "0"
  M_prime_vec.append(expr)

# Reshape M' to 3x3
M_prime = [M_prime_vec[i * 3 : (i + 1) * 3] for i in range(3)]

# Print nicely
print("Transformed matrix M':")
for i, row in enumerate(M_prime):
  print(f"Row {i}: {row}")
