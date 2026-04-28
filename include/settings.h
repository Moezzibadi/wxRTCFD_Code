#ifndef SETTINGS_H_INCLUDED
#define SETTINGS_H_INCLUDED

#ifdef USE_WX
#include <wx/wx.h>
#else
#include <string>
#include <vector>
struct wxPoint {
    int x, y;
    wxPoint() : x(0), y(0) {}
    wxPoint(int _x, int _y) : x(_x), y(_y) {}
};
struct wxArrayString {
    void Add(const std::string& s) {}
};
#endif

#include <vector>
#include <algorithm>

using namespace std;

// Structure pour représenter un point en 2D
struct RTCFDPoint {
    double x;
    double y;
};

wxArrayString getCaseList();
wxArrayString getObstacleList();
wxArrayString getScalarList();

vector<wxPoint> getSquarePoints(wxPoint pos, double length);
vector<RTCFDPoint> getSquarePoints(RTCFDPoint pos, double length);

vector<wxPoint> getDiamondPoints(wxPoint pos, double length);
vector<RTCFDPoint> getDiamondPoints(RTCFDPoint pos, double length);

vector<wxPoint> getNacaPoints(wxPoint pos, double length);
vector<RTCFDPoint> getNacaPoints(RTCFDPoint pos, double length); 
vector<RTCFDPoint> generateNacaProfile(RTCFDPoint pos, double chord, double thickness, int nb_points, double incidence=M_PI/12);
vector<wxPoint> generateNacaProfile(wxPoint pos, double chord, double thickness, int nb_points, double incidence=M_PI/12);

vector<vector<RTCFDPoint>> generateRotorPoints(RTCFDPoint pos, double length);
vector<vector<wxPoint>> generateRotorPoints(wxPoint pos, double length);
vector<vector<RTCFDPoint>> generateRotor(RTCFDPoint center, double radius, double chord, double thickness, int nb_points, int Z);
vector<vector<wxPoint>> generateRotor(wxPoint center, double radius, double chord, double thickness, int nb_points, int Z);

wxPoint *fromVectorToPtr(vector<wxPoint> pt);

bool isInsidePolygon(vector<RTCFDPoint> polygon, RTCFDPoint P);
vector<RTCFDPoint> rotatePolygon(vector<RTCFDPoint> polygon, RTCFDPoint center, double theta);
vector<wxPoint> rotatePolygon(vector<wxPoint> polygon, wxPoint center, double theta);
vector<vector<RTCFDPoint>> generateCircularRepeats(vector<RTCFDPoint> polygon, RTCFDPoint center, int n);

#endif // SETTINGS_H_INCLUDED
