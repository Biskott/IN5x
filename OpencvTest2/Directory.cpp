#include "stdafx.h"
#include "Directory.h"

vector<string> getFile(string value)
{
	vector<string> fileList = vector<string>();
	cout << "coucou" << endl;
	WIN32_FIND_DATA search_data;

	memset(&search_data, 0, sizeof(WIN32_FIND_DATA));

	HANDLE handle = FindFirstFile(LPCWSTR(value.c_str()), &search_data);

	while (handle != INVALID_HANDLE_VALUE)
	{
		printf("Found file: %s\r\n", search_data.cFileName);
		cout << "coucou" << endl;
		wstring ws(search_data.cFileName);
		fileList.push_back(string(ws.begin(), ws.end()));

		if (FindNextFile(handle, &search_data) == FALSE)
			break;
	}
	FindClose(handle);

	return fileList;
}