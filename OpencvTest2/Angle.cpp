#include "stdafx.h"
#include "Angle.h"

float angleConversion(float angle) {

	if (angle < -45)
		return 90 + angle;
	else
		return angle;
}