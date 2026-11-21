"""Check how many derivatives identify the frequencies on the constructed menus."""

from argparse import ArgumentParser
from fractions import Fraction
import json
from pathlib import Path
import sys


def matrix_rank(rows, cols):
    basis = {}
    for row in rows:
        row = list(row)
        for pivot, vector in sorted(basis.items()):
            if row[pivot]:
                multiplier = row[pivot]
                row = [a-multiplier*b for a, b in zip(row, vector, strict=True)]
        lead = next((j for j, value in enumerate(row) if value), None)
        if lead is not None:
            basis[lead] = [v/row[lead] for v in row]
            if len(basis) == cols:
                break
    return len(basis)


def full_modular_column_rank(rows, columns, prime=1_000_000_007):
    matrix = [
        [x.numerator*pow(x.denominator, -1, prime) % prime for x in row]
        for row in rows
    ]
    for column in range(columns):
        pivot = next(i for i in range(column, len(matrix)) if matrix[i][column])
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        leading = matrix[column]
        inverse = pow(leading[column], -1, prime)
        for j in range(column + 1, len(matrix)):
            scale = matrix[j][column]*inverse % prime
            matrix[j] = [(a-scale*b) % prime for a, b in zip(matrix[j], leading, strict=True)]
    return True


def evaluate(cases, order):
    magnitudes, _ = cases.independent_magnitudes(cases.derivative_polynomials(order))
    signed = sorted(magnitudes + [-x for x in magnitudes])
    # The general fixed-support guarantee requires r >= number of menus - 1.
    max_order = len(signed) - 1
    polys = cases.derivative_polynomials(max_order)
    for symmetric, support in ((False, signed), (True, magnitudes)):
        rows = [[Fraction(1)]*len(support)]
        recorded = []
        for r in range(2, max_order + 1):
            derivative = [cases.poly_at(polys[r], cases.oracle(x)) for x in support]
            assert all(derivative)
            rows.extend(tuple(d*x**j for d, x in zip(derivative, support, strict=True))
                        for j in range(r+1) if not symmetric or (r+j)%2==0)
            rank = matrix_rank(rows, len(support))
            recorded.append([r, rank])
            if rank == len(support):
                break
        indices = {n: rank for n, rank in recorded}
        assert rank == len(support)
        assert full_modular_column_rank(rows, len(support))
        lower_order = order + 1 if symmetric else order + 2
        if symmetric:
            usable_even_rows = 1 + sum(k//2 + 1 for k in range(2, order + 1))
            assert usable_even_rows == len(magnitudes)-1
        else:
            usable_odd_rows = sum((k+1)//2 for k in range(2, order + 2))
            assert usable_odd_rows == len(magnitudes)-1
        assert r == lower_order
        if symmetric and order >= 2:
            assert indices[order] == len(support)-1
        yield dict(matched_order=order, frequency_class="symmetric" if symmetric else "unrestricted",
                   magnitudes=len(magnitudes), signed_menus=len(signed),
                   rank_at_matched_order=1 if order == 1 else indices[order],
                   first_full_rank_order=r, parity_lower_bound_order=lower_order,
                   full_rank_verified_modular=True, ranks=recorded)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("oracle_designs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.oracle_designs.resolve()))
    import verify as cases

    results = [record for order in range(1, 10) for record in evaluate(cases, order)]
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    for row in results:
        print(row["frequency_class"], row["matched_order"], row["signed_menus"],
              row["rank_at_matched_order"], row["first_full_rank_order"])


if __name__ == "__main__":
    main()
