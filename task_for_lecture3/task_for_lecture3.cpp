#include <stdio.h>
#include <ctime>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <iostream>
#include <chrono>

using namespace std::chrono;

// количество строк в исходной квадратной матрице
const int MATRIX_SIZE = 1500;

/// Функция InitMatrix() заполняет переданную в качестве 
/// параметра квадратную матрицу случайными значениями
/// matrix - исходная матрица СЛАУ
void InitMatrix(double** matrix)
{
	for (int i = 0; i < MATRIX_SIZE; ++i)
	{
		matrix[i] = new double[MATRIX_SIZE + 1];
	}

	for (int i = 0; i < MATRIX_SIZE; ++i)
	{
		for (int j = 0; j <= MATRIX_SIZE; ++j)
		{
			matrix[i][j] = rand() % 2500 + 1;
		}
	}
}

/// Функция CheckResult() провеяет правильность полученного решения
/// путем вычисления нормы A*X - B
/// matrix - исходная матрица СЛАУ
/// последний столбец матрицы - значения правых частей уравнений
/// rows - количество строк в исходной матрице
/// result - массив ответов СЛАУ
void CheckResult(double **matrix, const int rows, double *result)
{
	double norm = 0.0;

	for (int i = 0; i < rows; ++i)
	{
		double sum = 0.0;
		for (int j = 0; j < rows; ++j)
			sum += matrix[i][j] * result[j];

		sum = std::abs(sum - matrix[i][rows]);

		if (norm < sum)
			norm = sum;
	}
	std::cout << "Result is: ||A*X - B|| = " << norm << std::endl;
}

/// Функция CopyMatrix() копирует матрицу src в dst
/// rows - число строк в матрице src
void CopyMatrix(double **src, double **dst, const int rows)
{
	for (int i = 0; i < rows; ++i)
	{
		dst[i] = new double[rows + 1];
	}
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j <= rows; ++j)
		{
			dst[i][j] = src[i][j];
		}
	}
}

/// Функция FreeMatrix() освобождает память, используемую
/// для хранения матрицы
/// matrix - матрица
/// rows - количество строк в матрице matrix
void FreeMatrix(double **matrix, const int rows)
{
	for (int i = 0; i < rows; ++i)
	{
		delete[]matrix[i];
	}
	delete[]matrix;
}

/// Функция SerialGaussMethod() решает СЛАУ методом Гаусса 
/// matrix - исходная матрица коэффиициентов уравнений, входящих в СЛАУ,
/// последний столбец матрицы - значения правых частей уравнений
/// rows - количество строк в исходной матрице
/// result - массив ответов СЛАУ
/// возвращает время выполнения прямого хода метода Гаусса
double SerialGaussMethod(double **matrix, const int rows, double* result)
{
	int k;
	double koef;

	high_resolution_clock::time_point t_start = high_resolution_clock::now();

	// прямой ход метода Гаусса
	for (k = 0; k < rows; ++k)
	{
		//
		for (int i = k + 1; i < rows; ++i)
		{
			koef = -matrix[i][k] / matrix[k][k];

			for (int j = k; j <= rows; ++j)
			{
				matrix[i][j] += koef * matrix[k][j];
			}
		}
	}

	high_resolution_clock::time_point t_end = high_resolution_clock::now();

	// обратный ход метода Гаусса
	result[rows - 1] = matrix[rows - 1][rows] / matrix[rows - 1][rows - 1];

	for (k = rows - 2; k >= 0; --k)
	{
		result[k] = matrix[k][rows];

		//
		for (int j = k + 1; j < rows; ++j)
		{
			result[k] -= matrix[k][j] * result[j];
		}

		result[k] /= matrix[k][k];
	}
	duration<double> duration = (t_end - t_start);
	return duration.count();
}

/// Функция ParallelGaussMethod() решает СЛАУ методом Гаусса 
/// matrix - исходная матрица коэффиициентов уравнений, входящих в СЛАУ,
/// последний столбец матрицы - значения правых частей уравнений
/// rows - количество строк в исходной матрице
/// result - массив ответов СЛАУ
/// возвращает время выполнения прямого хода метода Гаусса
double ParallelGaussMethod(double **matrix, const int rows, double* result)
{
	int k;

	high_resolution_clock::time_point t_start = high_resolution_clock::now();

	// прямой ход метода Гаусса
	for (k = 0; k < rows; ++k)
	{
		//
		cilk_for (int i = k + 1; i < rows; ++i)
		{
			double koef = -matrix[i][k] / matrix[k][k];

			for (int j = k; j <= rows; ++j)
			{
				matrix[i][j] += koef * matrix[k][j];
			}
		}
	}

	high_resolution_clock::time_point t_end = high_resolution_clock::now();

	// обратный ход метода Гаусса
	result[rows - 1] = matrix[rows - 1][rows] / matrix[rows - 1][rows - 1];

	for (k = rows - 2; k >= 0; --k)
	{
		//
		cilk::reducer_opadd<double> res(matrix[k][rows]);
		cilk_for(int j = k + 1; j < rows; ++j)
		{
			res -= matrix[k][j] * result[j];
		}

		result[k] = res->get_value() / matrix[k][k];
	}
	duration<double> duration = (t_end - t_start);
	return duration.count();
}

int main()
{
	srand((unsigned)time(0));

	int i;

	/*/
	// кол-во строк в матрице, приводимой в качестве примера
	const int test_matrix_lines = 4;

	double **test_matrix = new double*[test_matrix_lines];

	// цикл по строкам
	for (i = 0; i < test_matrix_lines; ++i)
	{
		// (test_matrix_lines + 1)- количество столбцов в тестовой матрице,
		// последний столбец матрицы отведен под правые части уравнений, входящих в СЛАУ
		test_matrix[i] = new double[test_matrix_lines + 1];
	}

	// массив решений СЛАУ
	double *result = new double[test_matrix_lines];

	// инициализация тестовой матрицы
	test_matrix[0][0] = 2; test_matrix[0][1] = 5;  test_matrix[0][2] = 4;  test_matrix[0][3] = 1;  test_matrix[0][4] = 20;
	test_matrix[1][0] = 1; test_matrix[1][1] = 3;  test_matrix[1][2] = 2;  test_matrix[1][3] = 1;  test_matrix[1][4] = 11;
	test_matrix[2][0] = 2; test_matrix[2][1] = 10; test_matrix[2][2] = 9;  test_matrix[2][3] = 7;  test_matrix[2][4] = 40;
	test_matrix[3][0] = 3; test_matrix[3][1] = 8;  test_matrix[3][2] = 9;  test_matrix[3][3] = 2;  test_matrix[3][4] = 37;

	SerialGaussMethod(test_matrix, test_matrix_lines, result);

	for (i = 0; i < test_matrix_lines; ++i)
	{
		delete[]test_matrix[i];
	}

	printf("Solution:\n");

	for (i = 0; i < test_matrix_lines; ++i)
	{
		printf("x(%d) = %lf\n", i, result[i]);
	}

	delete[] result;
	/**/

	double **matrix = new double*[MATRIX_SIZE];
	double **matrix_serial = new double*[MATRIX_SIZE];
	double **matrix_parallel = new double*[MATRIX_SIZE];
	double *result = new double[MATRIX_SIZE];

	InitMatrix(matrix);
	CopyMatrix(matrix, matrix_serial, MATRIX_SIZE);
	CopyMatrix(matrix, matrix_parallel, MATRIX_SIZE);

	double time_serial = SerialGaussMethod(matrix_serial, MATRIX_SIZE, result);
	std::cout << "Time in serial variant is: " << time_serial << " sec" << std::endl;
	CheckResult(matrix, MATRIX_SIZE, result);
	
	double time_parallel = ParallelGaussMethod(matrix_parallel, MATRIX_SIZE, result);
	std::cout << "Time in parallel variant is: " << time_parallel << " sec" << std::endl;
	CheckResult(matrix, MATRIX_SIZE, result);

	double acceleration = time_serial / time_parallel;
	std::cout << "Acceleration is: " << acceleration << std::endl;

	FreeMatrix(matrix, MATRIX_SIZE);
	FreeMatrix(matrix_serial, MATRIX_SIZE);
	FreeMatrix(matrix_parallel, MATRIX_SIZE);
	delete[]result;

	return 0;
}