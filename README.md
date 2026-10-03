# Report 1 - 역행렬 계산 프로그램
# Discrete Mathematics / 국민대학교
# Methods:
# 1) 행렬식을 이용한 역행렬 계산 (cofactor/adjugate)
# 2) 가우스-조던 소거법을 이용한 역행렬 계산
# Additional feature:
# 3) 계산된 역행렬이 맞는지 A * A^(-1) = I 로 자동 검증

EPS = 1e-10


def print_matrix(matrix, title=""):
    """행렬을 보기 쉽게 출력한다."""
    if title:
        print(f"\n[{title}]")
    for row in matrix:
        print(" ".join(f"{x:10.4f}" for x in row))


def input_matrix():
    """n x n 정수 행렬을 행 단위로 입력받아 2차원 리스트로 저장한다."""
    while True:
        try:
            n = int(input("행렬의 크기 n을 입력하세요 (n x n): "))
            if n <= 0:
                print("오류: n은 1 이상의 정수여야 합니다.")
                continue
            break
        except ValueError:
            print("오류: 정수를 입력하세요.")

    matrix = []
    print(f"{n}개의 정수를 한 줄에 입력하세요.")

    for i in range(n):
        while True:
            try:
                row = list(map(int, input(f"{i + 1}행 입력: ").split()))
                if len(row) != n:
                    print(f"오류: 정확히 {n}개의 정수를 입력하세요.")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("오류: 정수만 입력하세요.")

    return matrix


def determinant(matrix):
    """Laplace 전개를 재귀적으로 사용하여 행렬식을 계산한다."""
    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        minor = [
            [matrix[i][k] for k in range(n) if k != j]
            for i in range(1, n)
        ]
        sign = 1 if j % 2 == 0 else -1
        det += sign * matrix[0][j] * determinant(minor)

    return det


def minor_matrix(matrix, row_to_remove, col_to_remove):
    """지정한 행과 열을 제거한 소행렬을 만든다."""
    return [
        [matrix[i][j] for j in range(len(matrix)) if j != col_to_remove]
        for i in range(len(matrix))
        if i != row_to_remove
    ]


def inverse_by_determinant(matrix):
    """행렬식 + 여인수행렬/수반행렬 방법으로 역행렬을 계산한다."""
    n = len(matrix)
    det = determinant(matrix)

    if abs(det) < EPS:
        raise ValueError("행렬식이 0이므로 역행렬이 존재하지 않습니다.")

    # 1x1 행렬은 별도로 처리
    if n == 1:
        return [[1.0 / matrix[0][0]]]

    # Cofactor matrix
    cofactors = []
    for i in range(n):
        row = []
        for j in range(n):
            minor = minor_matrix(matrix, i, j)
            sign = 1 if (i + j) % 2 == 0 else -1
            row.append(sign * determinant(minor))
        cofactors.append(row)

    # adj(A) = cofactor matrix의 transpose
    adjugate = [list(row) for row in zip(*cofactors)]

    return [[adjugate[i][j] / det for j in range(n)] for i in range(n)]


def inverse_by_gauss_jordan(matrix):
    """가우스-조던 소거법으로 역행렬을 계산한다."""
    n = len(matrix)

    # [A | I] 형태의 확대행렬
    augmented = [
        [float(x) for x in matrix[i]] +
        [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]

    for col in range(n):
        # 현재 열에서 가장 큰 pivot을 선택한다.
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))

        if abs(augmented[pivot_row][col]) < EPS:
            raise ValueError("역행렬이 존재하지 않습니다. (pivot = 0)")

        # pivot 행을 현재 행으로 이동
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = (
                augmented[pivot_row], augmented[col]
            )

        # pivot을 1로 만든다.
        pivot = augmented[col][col]
        augmented[col] = [x / pivot for x in augmented[col]]

        # 현재 열의 다른 모든 값을 0으로 만든다.
        for r in range(n):
            if r == col:
                continue

            factor = augmented[r][col]
            if abs(factor) < EPS:
                continue

            augmented[r] = [
                augmented[r][c] - factor * augmented[col][c]
                for c in range(2 * n)
            ]

    return [row[n:] for row in augmented]


def multiply_matrices(a, b):
    """두 행렬을 곱한다."""
    rows = len(a)
    cols = len(b[0])
    inner = len(b)

    return [
        [
            sum(a[i][k] * b[k][j] for k in range(inner))
            for j in range(cols)
        ]
        for i in range(rows)
    ]


def is_identity(matrix):
    """행렬이 단위행렬인지 확인한다."""
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            if abs(matrix[i][j] - expected) > 1e-7:
                return False
    return True


def compare_matrices(a, b):
    """두 역행렬 결과가 같은지 비교한다."""
    n = len(a)
    max_error = 0.0

    for i in range(n):
        for j in range(n):
            max_error = max(max_error, abs(a[i][j] - b[i][j]))

    return max_error <= 1e-7, max_error


def main():
    print("=" * 60)
    print("          역행렬 계산 프로그램")
    print("=" * 60)

    matrix = input_matrix()

    print_matrix(matrix, "입력 행렬 A")

    # 행렬식 계산
    det = determinant(matrix)
    print(f"\n행렬식 det(A) = {det}")

    if abs(det) < EPS:
        print("\n[오류]")
        print("행렬식이 0이므로 역행렬이 존재하지 않습니다.")
        print("두 방법의 역행렬 계산을 수행할 수 없습니다.")
        return

    try:
        # 방법 1: 행렬식
        inverse_det = inverse_by_determinant(matrix)

        # 방법 2: Gauss-Jordan
        inverse_gj = inverse_by_gauss_jordan(matrix)

        print_matrix(
            inverse_det,
            "방법 1: 행렬식을 이용한 역행렬"
        )

        print_matrix(
            inverse_gj,
            "방법 2: 가우스-조던 소거법을 이용한 역행렬"
        )

        # 두 결과 비교
        same, max_error = compare_matrices(inverse_det, inverse_gj)

        print("\n[두 방법 결과 비교]")
        print(f"최대 오차 = {max_error:.10f}")

        if same:
            print("결과가 동일합니다. ✓")
        else:
            print("결과가 다릅니다. 다시 확인하세요.")

        # 추가 기능: A * A^(-1) = I 검증
        verification = multiply_matrices(matrix, inverse_gj)
        print_matrix(
            verification,
            "추가기능: A × A^(-1) 검증 결과"
        )

        if is_identity(verification):
            print("검증 성공: A × A^(-1) = I 입니다. ✓")
        else:
            print("검증 실패: 단위행렬이 아닙니다.")

    except ValueError as e:
        print(f"\n[오류] {e}")


if __name__ == "__main__":
    main()
