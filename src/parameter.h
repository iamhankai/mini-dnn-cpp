#ifndef _parameter_h_
#define _parameter_h_

#include <string>
#include <vector>
#include <fstream>

std::vector<float> loadParametersFromFile(std::string filename);
void storeParametersToFile(std::string filename, std::vector<float> parameters);

#endif