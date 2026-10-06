/*
 * To compile need to run
 * g++ -fopenmp -O2 multiCore.cpp -o multiCore
 */

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <omp.h>

using namespace std;

#define SYSTEMTIME clock_t

// single core, sequential matrix multiplication
void seq_mult(int m_ar, int m_br){

    char st[100];
    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phc = (double *)malloc((m_ar * m_ar) * sizeof(double));

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(int i = 0; i < m_br; i++)
        for(int j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    double startTime = omp_get_wtime();

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_br; j++){
            double temp = 0;
            for(int k = 0; k < m_ar; k++)    
                temp += pha[i*m_ar + k] * phb[k*m_br + j];
            phc[i*m_ar + j] = temp;
        }

    double endTime = omp_get_wtime();

    snprintf(st, sizeof(st), "Time: %3.3f seconds\n",
         (double)(endTime - startTime));
    
    cout << st;
    cout << "Result matrix: " << endl;
    
    for(int i = 0; i < 1; i++){
        for(int j = 0; j < min(10, m_br); j++)
            cout << phc[j] << " ";
    }
    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}

// multi-core, parallelism in outer loop matrix multiplication
void par_outer_mult(int m_ar, int m_br){
        
    char st[100];
    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phc = (double *)malloc((m_ar * m_ar) * sizeof(double));

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(int i = 0; i < m_br; i++)
        for(int j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    double startTime = omp_get_wtime();
    
    #pragma omp parallel for
    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_br; j++){
            double temp = 0;
            for(int k = 0; k < m_ar; k++)    
                temp += pha[i*m_ar + k] * phb[k*m_br + j];
            phc[i*m_ar + j] = temp;
        }

    double endTime = omp_get_wtime();

    snprintf(st, sizeof(st), "Time: %3.3f seconds\n",
         (double)(endTime - startTime));
    
    cout << st;
    cout << "Result matrix: " << endl;
    
    for(int i = 0; i < 1; i++){
        for(int j = 0; j < min(10, m_br); j++)
            cout << phc[j] << " ";
    }
    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}

// multi-core, parallelism in inner loop matrix multiplication
void par_inner_mult(int m_ar, int m_br){

    char st[100];
    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phc = (double *)malloc((m_ar * m_ar) * sizeof(double));

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(int i = 0; i < m_br; i++)
        for(int j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    double startTime = omp_get_wtime();
    
    #pragma omp parallel
    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_br; j++){
            double temp = 0;
            #pragma omp for 
            for(int k = 0; k < m_ar; k++)    
                temp += pha[i*m_ar + k] * phb[k*m_br + j];
            phc[i*m_ar + j] = temp;
        }

    double endTime = omp_get_wtime();

    snprintf(st, sizeof(st), "Time: %3.3f seconds\n",
         (double)(endTime - startTime));
    
    cout << st;
    cout << "Result matrix: " << endl;
    
    for(int i = 0; i < 1; i++){
        for(int j = 0; j < min(10, m_br); j++)
            cout << phc[j] << " ";
    }
    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}


// sequential line-by-line matrix multiplication
void seq_line_mult(int m_ar, int m_br){
    char st[100];
    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_br) * sizeof(double));
    phc = (double *)malloc((m_ar * m_br) * sizeof(double));

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_br; j++)
            phc[i*m_br + j] = 0.0;

    double startTime = omp_get_wtime();

    for(int i = 0; i < m_ar; i++){
        for(int k = 0; k < m_ar; k++){
            for(int j = 0; j < m_br; j++){
                phc[i*m_br + j] += pha[i*m_ar + k] * phb[k*m_br + j];
            }
        }
    }

    double endTime = omp_get_wtime();

    snprintf(st, sizeof(st), "Time: %3.3f seconds\n",
            (double)(endTime - startTime));
    
    cout << st;
    cout << "Result matrix: " << endl;
    
    for(int i = 0; i < 1; i++){
        for(int j = 0; j < min(10, m_br); j++)
            cout << phc[j] << " ";
    }

    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}


// multi-core, parallel line-by-line matrix multiplication
void par_line_mult(int m_ar, int m_br){
    char st[100];
    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_br) * sizeof(double));
    phc = (double *)malloc((m_ar * m_br) * sizeof(double));

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_br; j++)
            phc[i*m_br + j] = 0.0;

    double startTime = omp_get_wtime();
    
    #pragma omp parallel for
    for(int i = 0; i < m_ar; i++){
        for(int k = 0; k < m_ar; k++){
            for(int j = 0; j < m_br; j++){
                phc[i*m_br + j] += pha[i*m_ar + k] * phb[k*m_br + j];
            }
        }
    }

    double endTime = omp_get_wtime();

    snprintf(st, sizeof(st), "Time: %3.3f seconds\n",
            (double)(endTime - startTime));
    
    cout << st;
    cout << "Result matrix: " << endl;
    
    for(int i = 0; i < 1; i++){
        for(int j = 0; j < min(10, m_br); j++)
            cout << phc[j] << " ";
    }

    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}

// multi-core, parallel and SIMD line-by-line matrix multiplication
void par_simd_line_mult(int m_ar, int m_br){
    char st[100];
    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_br) * sizeof(double));
    phc = (double *)malloc((m_ar * m_br) * sizeof(double));

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_br; j++)
            phc[i*m_br + j] = 0.0;

    double startTime = omp_get_wtime();

    #pragma omp parallel for
    for(int i = 0; i < m_ar; i++){
        for(int k = 0; k < m_ar; k++){
            #pragma omp simd
            for(int j = 0; j < m_br; j++){
                phc[i*m_br + j] += pha[i*m_ar + k] * phb[k*m_br + j];
            }
        }
    }

    double endTime = omp_get_wtime();

    snprintf(st, sizeof(st), "Time: %3.3f seconds\n",
            (double)(endTime - startTime));
    
    cout << st;
    cout << "Result matrix: " << endl;
    
    for(int i = 0; i < 1; i++){
        for(int j = 0; j < min(10, m_br); j++)
            cout << phc[j] << " ";
    }

    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}

// multi-core, parallel and colapsed loop line-by-line matrix multiplication
void par_collapse_line_mult(int m_ar, int m_br){
    char st[100];
    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_br) * sizeof(double));
    phc = (double *)malloc((m_ar * m_br) * sizeof(double));

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_br; j++)
            phc[i*m_br + j] = 0.0;

    double startTime = omp_get_wtime();
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < m_ar; i++){
        for(int k = 0; k < m_ar; k++){
            for(int j = 0; j < m_br; j++){
                phc[i*m_br + j] += pha[i*m_ar + k] * phb[k*m_br + j];
            }
        }
    }

    double endTime = omp_get_wtime();

    snprintf(st, sizeof(st), "Time: %3.3f seconds\n",
            (double)(endTime - startTime));
    
    cout << st;
    cout << "Result matrix: " << endl;
    
    for(int i = 0; i < 1; i++){
        for(int j = 0; j < min(10, m_br); j++)
            cout << phc[j] << " ";
    }

    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}


int main(int argc, char *argv[]){
    int lin, col;
    int op;

    do {
        cout << endl;
        cout << "1. Sequential Multiplication " << endl;
        cout << "2. <#pragma omp for> in outer loop parallel Multiplication" << endl;
        cout << "3. <#pragma omp for> in inner loop parallel Multiplication" << endl; 
        cout << "4. Sequencial Line Multiplication " << endl;
        cout << "5. Parallel Line Multiplication " << endl;
        cout << "6. Parallel + SIMD Line Multiplication" << endl;
        cout << "7. Parallel + collapse(2) Line Multiplication" << endl;
        cout << "0. Exit" << endl;

        cout << "Selection?: ";
        cin >> op;

        if (op == 0)
            break;

        cout << "Dimensions: lins=cols ? ";
        cin >> lin;
        col = lin;

        switch (op) {
            case 1:
                seq_mult(lin, col);
                break;
            case 2:
                par_outer_mult(lin, col);
                break;
            case 3:
                par_inner_mult(lin, col);
                break;
            case 4:
                seq_line_mult(lin, col);
                break;
            case 5:
                par_line_mult(lin,col);
                break;
            case 6:
                par_simd_line_mult(lin,col);
                break;
            case 7:
                par_collapse_line_mult(lin, col);
                break;
        }

    } while (op != 0);

    return 0;
}
