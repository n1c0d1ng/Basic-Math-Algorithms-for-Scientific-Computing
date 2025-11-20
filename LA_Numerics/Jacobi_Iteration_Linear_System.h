/*/
Solving_Linear_System mit Hilfe des Jacobi Verfahrens
---------------------------------------------------------------------------------------------------------------------
x_{i+1} = D^{-1} (b + (L+U) x_{i})
Iteratives Verfahren => Konvergenz für symmetrische positiv definite Matrizen
Konvergenzbeweis mit Hilfe des Banachschen Fixpunktsatzes
------------------------------------------------------------------------------------------------------------------------------
/*/

double *solveLinearSystem(double **A, double *b, double *x_start ,long anzZeilen, double epsilon)
{
// Definition der entsprechenden Variablen
//---------------------------------------------------------------------------------------------------------------------
  int iterations = 0;
  long maxIterations = 1000;
  long i,j;
  double *x_alt = NULL;
  double *x_neu = NULL;
  double *b_approx = NULL;
  double norm = epsilon + 1;
  
  x_neu = vector_neu(anzZeilen);
  x_alt = vector_neu(anzZeilen);
  vector_kopieren(x_alt, x_start, anzZeilen);
  
// Anwenden des iterativen Verfahrens
//---------------------------------------------------------------------------------------------------------------------
  while ((norm > epsilon) && (iterations < maxIterations)) {
    
// Berechnung des Updates
//---------------------------------------------------------------------------------------------------------------------
    for (i=0; i < anzZeilen; i++) {
      for (j=0; j < anzZeilen; j++) {
        if (j != i) {
          x_neu[i] = b[i] - A[i][j] * x_alt[j];
        }
      }
      x_neu[i] = x_neu[i] / A[i][i];
    }
    
// Berechnung der Norm der Differenz des Updates
//---------------------------------------------------------------------------------------------------------------------
    norm = 0;
    for (i=0; i < anzZeilen; i++) {
      norm += pow(x_neu[i] - x_alt[i], 2);
    }

// Initialisieren der Werte
//---------------------------------------------------------------------------------------------------------------------
    norm = sqrt(norm);
    printf("Die Norm beträgt %lf \n" , norm);
    iterations += 1;
    printf("Neuer Vektor \n");
    vector_ausgeben(x_neu, anzZeilen);
    b_approx = matrix_vector_prod(A, x_neu, anzZeilen, anzZeilen);
    printf("Kontrollrechnung zeigt \n");
    printf("Ax= \n");
    vector_ausgeben(b_approx, 2);
    vector_kopieren(x_alt, x_neu, anzZeilen);
    printf("-----------------------------------------------------------------------------------------------------------\n");
  }

  printf("Anzahl Iterationen %d " , iterations);
  return x_neu;
}
