/*
Allgemeine Matrix Operationen
---------------------------------------------------------------------------------------------------------------------
*/


// Matrix anlegen
double **matrix_neu(long m,long n)
{
  // Anzahl der Spalten m, Anzahl der Zeilen n
  long i;
  double **A;
  
  // WICHTIG: sizeof(double *)
  A = (double **) calloc( m, sizeof(double *) );
  
  if (A == NULL) {  // Fehlerbehandlung
    fprintf(stderr, "Fehler beim anlegen einer Matrix.\n");
    exit(1); // Programm mit Rueckgabewert 1 beenden!!
  }
  
  // Lege die einzelnen Zeilen an: Zugriff => A[i][j]
  for (i=0; i<m; i++) {
    A[i] = (double *) calloc( n, sizeof(double) );
    
    if (A[i] == NULL) {  // Fehlerbehandlung
      fprintf(stderr, "Fehler beim anlegen einer Matrix.\n");
      exit(1); // Programm mit Rueckgabewert 1 beenden!!
    }
  }
  return A;
}

// Matrix anlegen
double *vector_neu(long m)
{
  // Anzahl der Zeilen m
  double *v;
  
  // WICHTIG: sizeof(double *)
  v = (double *) calloc( m, sizeof(double));
  
  if (v == NULL) {  // Fehlerbehandlung
    fprintf(stderr, "Fehler beim anlegen einer Matrix.\n");
    exit(1); // Programm mit Rueckgabewert 1 beenden!!
  }
  return v;
}

//Matrix ausgeben
void matrix_ausgeben(double **M, long m, long n)
{
  int k,l;
  
  for (k=0;k <= m-1; k++)
  {
    for (l=0;l <= n-1; l++)
    {
      printf("%lf ", M[k][l]);
    }
    printf("\n");
  }
}

// Vector ausgeben
void vector_ausgeben(double *M, long m)
{
  int k;
  
  for (k=0;k <= m-1; k++)
  {
    printf("%lf ", M[k]);
    printf("\n");
  }
}

//Speicher freigeben
void matrix_freigeben(double **A) {
  free(A);  // A = NULL; haette keinen Effekt, da Call by Value
}

double **matrix_einlesen(long anzZeilen, long anzSpalten) {
  int i,j;
  double **M;
  
  //printf("Bitte geben sie die Anzahl der Zeilen und Spalten ein: ");
  //scanf("%ld %ld", &anzZeilen, &anzSpalten);
  
  // Matrizen anlegen
  M = matrix_neu(anzZeilen, anzSpalten);
  
  // Matrix A mit Werten belegen
  for (i=0; i <= anzZeilen - 1; i++)
  {
    for (j=0; j <= anzSpalten - 1; j++)
    {
      printf ("Bitte Eintrag in Zeile %d und Spalte %d angeben: ", i+1, j+1);
      scanf("%lf", &M[i][j] );
    }
  }
  return M;
}

// Ziel A, Quelle B
void matrix_kopieren(double **A, const double **B, long anzZeilen, long anzSpalten)
  {
  long i, j;
  for (i=0; i<anzZeilen; i++)
    for (j=0; j<anzSpalten; j++)
      A[i][j] = B[i][j];
}

// Ziel A, Quelle B
void vector_kopieren(double *A, const double *B, long anzZeilen)
{
  long i;
  for (i=0; i < anzZeilen; i++)
  {
      A[i] = B[i];
  }
}


// Frobeniusnorm
double matrix_FrobeniusNorm(double **M, long m, long n) {
  double result = 0.0;
  int i,j;
  for(i=0; i < m; ++i)
  {
      for(j=0; j < n; ++j)
      {
        double value = M[i][j];
        result += value * value;
      }
   }
  return sqrt(result);
}

// Matrixaddition p*A+B -> B
long matrix_addieren(double p, double **A, double **B, long m, long n) {
  int i,j;
  for(i=0; i < m; ++i) {
    for(j=0; j < n; ++j) {
       B[i][j] += p*A[i][j];
    }
  }
  return **B;
}

// Matrixmultiplikation p*A*B+q*C -> C
long matrix_mult(double p, double q, double **A, double **B, double **C, long m, long n, long l) {
  int i,j,k;
  double **D;
  D = matrix_neu(m,n);
  for (i=0; i < m; i++) {
    for (j=0; j < n; j++) {
      for (k=0; k < l; k++) {
        D[i][j] = A[i][k] *  B[k][j];
      }
    }
  }
  
  for(i=0; i < m; ++i) {
    for(j=0; j < n; ++j) {
      C[i][j] *= q;
    }
  }
  return matrix_addieren(p, D, C, m, n);
}

// Matrix_Vektor Produkt
// A ist mxn und v ist nx1
double  *matrix_vector_prod(double **A, double *v, long m, long n) {
  long i,j;
  double *D;
  D = vector_neu(m);
  
  for (i=0; i<=m-1; i++) {
    for (j=0; j<= n-1; j++) {
      D[i] += A[i][j]*v[j];
    }
  }
  return D;
}

double norm_vector(double *x, long anzZeilen) {
  double norm = 0;
  long i;
  
  for (i=0; i<anzZeilen; i++) {
    norm += pow(x[i], 2);
  }
  return sqrt(norm);
}
