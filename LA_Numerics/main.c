/*
Solving_Linear_System
Created by Nicolas Schäfer
-----------------------------------------------------------------------------------------------------------------------
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix_funktionen.h"
#include "Jacobi_Iteration_Linear_System.h"

int main(int argc, const char * argv[]) {
  
// Definieren der entsprechenden Arrays
//---------------------------------------------------------------------------------------------------------------------
  double **A;
  double *b;
  double *x;
  double *x_start;

// Anlegen des Speichers für die benoetigten Arrays
//---------------------------------------------------------------------------------------------------------------------
  A = matrix_neu(2, 2);
  b = vector_neu(2);
  x_start =vector_neu(2);
  x = vector_neu(2);

// Beispielhafte statische Dateneingabe
//---------------------------------------------------------------------------------------------------------------------
  A[0][0] = 2;
  A[0][1] = 1;
  A[1][0] = 3;
  A[1][1] = 4;
  b[0] = 1;
  b[1] = 2;
  x_start[0] = 1;
  x_start[1] = 1;
  
// Aufrufen des entsprechenden Gleichungssystem Lösers und Ausgabe der Werte
//---------------------------------------------------------------------------------------------------------------------
  x = solveLinearSystem(A,b,x_start, 2, 0.00001);
  printf("Lösung des Systems x = \n");
  vector_ausgeben(x, 2);
  
  return 0;
}
