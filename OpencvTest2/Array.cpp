#include "stdafx.h"
#include "Array.h"

void swap(Point2d tableau[], int a, int b) {
	Point2d temp = tableau[a];
	tableau[a] = tableau[b];
	tableau[b] = temp;
}

void sortTabByY(Point2d tableau[], int debut, int fin) {
	int left = debut - 1;
	int right = fin + 1;
	const double pivot = tableau[debut].y;

	if (debut >= fin)
		return;

	while (true)
	{
		do right--; while (tableau[right].y > pivot);
		do left++; while (tableau[left].y < pivot);

		if (left < right)
			swap(tableau, left, right);
		else break;
	}
	sortTabByY(tableau, debut, right);
	sortTabByY(tableau, right + 1, fin);
}

bool vectorContains(vector<int> tab, int value) {

	for (int tabValue : tab) {
		if (tabValue == value)
			return true;
	}
	return false;
}