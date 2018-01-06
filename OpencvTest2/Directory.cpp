#include "stdafx.h"
#include "Directory.h"

vector<string> getFile(string value)
{
	vector<string> fileList = vector<string>();
	WIN32_FIND_DATA search_data;

	memset(&search_data, 0, sizeof(WIN32_FIND_DATA));
	HANDLE handle = FindFirstFile(LPCWSTR(value.c_str()), &search_data);
	
	if (handle == INVALID_HANDLE_VALUE) printf(value.c_str());

	while (handle != INVALID_HANDLE_VALUE)
	{
		printf("Found file: %s\r\n", search_data.cFileName);
		wstring ws(search_data.cFileName);
		fileList.push_back(string(ws.begin(), ws.end()));

		if (FindNextFile(handle, &search_data) == FALSE)
			break;
	}
	FindClose(handle);

	return fileList;
}

vector<string> getFile2() {

	vector<string> file;

	for (int i = 1; i <= 31; ++i) {
		stringstream ss;
		ss << i;
		file.push_back(ss.str() + "_picture.png");
		cout << file[i - 1] << endl;
	}
	return file;
}