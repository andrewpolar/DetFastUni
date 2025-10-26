#pragma once
#include "Helper.h"
#include "Univariate.h"
#include <memory>

class Urysohn
{
public:
	Urysohn(const std::vector<double>& xmin, const std::vector<double>& xmax,
		double targetMin, double targetMax, int points, int len) {
		_length = len;
		double ymin = targetMin / _length;
		double ymax = targetMax / _length;
		Helper::Sum2IndividualLimits(targetMin, targetMax, len, ymin, ymax);
		_univariateList = std::vector<std::unique_ptr<Univariate>>(_length);
		for (int i = 0; i < _length; ++i) {
			_univariateList[i] = std::make_unique<Univariate>(xmin[i], xmax[i], ymin, ymax, points);
		}
	}
	Urysohn(const Urysohn& uri) {
		_length = uri._length;
		_univariateList = std::vector<std::unique_ptr<Univariate>>(_length);
		for (int i = 0; i < _length; ++i) {
			_univariateList[i] = std::make_unique<Univariate>(*uri._univariateList[i]);
		}
	}
	void UpdateUsingInput(double delta, const std::vector<double>& inputs) {
		for (int i = 0; i < _length; ++i) {
			_univariateList[i]->UpdateUsingInput(inputs[i], delta);
		}
	}
	void UpdateUsingMemory(double delta) {
		for (int i = 0; i < _length; ++i) {
			_univariateList[i]->UpdateUsingMemory(delta);
		}
	}
	double GetValueUsingInput(const std::vector<double>& inputs, bool noUpdate = false) {
		double f = 0.0;
		for (int i = 0; i < _length; ++i) {
			f += _univariateList[i]->GetFunctionUsingInput(inputs[i], noUpdate);
		}
		return f;
	}
	void IncrementInner() {
		for (int i = 0; i < _length; ++i) {
			_univariateList[i]->IncrementPoints();
		}
	}
	std::vector<double> GetUPoints(int n) {
		return _univariateList[n]->GetAllPoints();
	}
	//functions for concurrent processing 
	void GetAllLastValues(std::vector<std::pair<int, double>>& allPairs) {
		for (int i = 0; i < _length; ++i) {
			_univariateList[i]->GetLastValues(allPairs[i]);
		}
	}
	void GetAllLeftValues(const std::vector<int>& pos, std::vector<double>& v) {
		for (int i = 0; i < _length; ++i) {
			_univariateList[i]->GetFunctionDirect(pos[i], v[i]);
		}
	}
	void GetAllRightValues(const std::vector<int>& pos, std::vector<double>& v) {
		for (int i = 0; i < _length; ++i) {
			_univariateList[i]->GetFunctionDirect(pos[i] + 1, v[i]);
		}
	}
	void SetLastValues(const std::vector<std::pair<int, double>>& Pairs) {
		for (int i = 0; i < _length; ++i) {
			_univariateList[i]->SetLastValues(Pairs[i]);
		}
	}
	void RescaleUrysohn(const Urysohn& uri, int n) {
		for (int i = 0; i < _length; ++i) {
			_univariateList[i]->RescaleFunction(uri._univariateList[i]->_y, n);
		}
	}
	std::vector<std::unique_ptr<Univariate>> _univariateList;
private:
	int _length;
};

