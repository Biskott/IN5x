#include "stdafx.h"
#include "Array.h"

bool vectorContains(vector<int> tab, int value) {

	for (int tabValue : tab) {
		if (tabValue == value)
			return true;
	}
	return false;
}