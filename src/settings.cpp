#include "settings.h"

wxArrayString getCaseList()
{
    wxArrayString arrCase;
    arrCase.Add("Tank");
    arrCase.Add("Wind tunnel");
    arrCase.Add("Paint");
    arrCase.Add("Wind tunnel (high resolution)");

    return arrCase;
}

wxArrayString getObstacleList()
{
    wxArrayString arrObject;
    arrObject.Add("Cylinder");
    arrObject.Add("Square");
    arrObject.Add("Diamond");
    arrObject.Add("NACA");
    arrObject.Add("Rotor");

    return arrObject;
}

wxArrayString getScalarList()
{
    wxArrayString arrScalar;
    arrScalar.Add("None");
    arrScalar.Add("Pressure");
    arrScalar.Add("Velocity");
    arrScalar.Add("x-Velocity");
    arrScalar.Add("y-Velocity");

    return arrScalar;
}

vector<wxPoint> getSquarePoints(wxPoint pos, double length)
{
    return {wxPoint(pos.x - length, pos.y + length),
            wxPoint(pos.x + length, pos.y + length),
            wxPoint(pos.x + length, pos.y - length),
            wxPoint(pos.x - length, pos.y - length)};
}

vector<RTCFDPoint> getSquarePoints(RTCFDPoint pos, double length)
{
    return {{pos.x - length, pos.y + length},
            {pos.x + length, pos.y + length},
            {pos.x + length, pos.y - length},
            {pos.x - length, pos.y - length}};
}

vector<wxPoint> getDiamondPoints(wxPoint pos, double length)
{
    double c = sqrt(2.0);
    return {wxPoint(pos.x, pos.y + c * length),
            wxPoint(pos.x + c * length, pos.y),
            wxPoint(pos.x, pos.y - c * length),
            wxPoint(pos.x - c * length, pos.y)};
}

vector<RTCFDPoint> getDiamondPoints(RTCFDPoint pos, double length)
{
    double c = sqrt(2.0);
    return {{pos.x, pos.y + c * length},
            {pos.x + c * length, pos.y},
            {pos.x, pos.y - c * length},
            {pos.x - c * length, pos.y}};
}

vector<wxPoint> getNacaPoints(wxPoint pos, double length)
{
    return generateNacaProfile(pos, 4.0 * length, 0.12, 10, M_PI / 12);
}

vector<RTCFDPoint> getNacaPoints(RTCFDPoint pos, double length)
{
    return generateNacaProfile(pos, 4.0 * length, 0.12, 10, -M_PI / 12);
}

wxPoint *fromVectorToPtr(vector<wxPoint> pt)
{
    wxPoint *ptr = new wxPoint[pt.size()];
    for (size_t i = 0; i < pt.size(); i++)
        ptr[i] = pt[i];
    return ptr;
}

bool isInsidePolygon(vector<RTCFDPoint> polygon, RTCFDPoint P)
{
    int count = 0;
    for (int i = 0; i < polygon.size(); i++)
    {
        RTCFDPoint A = polygon[i];
        RTCFDPoint B = polygon[(i + 1) % polygon.size()];

        if ((P.y > min(A.y, B.y)) && (P.y <= max(A.y, B.y)))
        {
            double x = A.x + (P.y - A.y) * (B.x - A.x) / (B.y - A.y);

            if (x > P.x)
            {
                count++;
            }
        }
    }
    return count & 1; //count % 2 == 1;
}

vector<RTCFDPoint> generateNacaProfile(RTCFDPoint pos, double chord, double thickness, int nb_points, double incidence)
{
    vector<RTCFDPoint> points;
    int contraction = 3;
    double c = chord;
    double t = thickness;
    int n = nb_points;

    for (int i = 0; i <= n; i++)
    {
        double x = c * pow(double(i) / n, contraction);
        double y = (c * t / 0.2) * (0.2969 * sqrt(x / c) - 0.1260 * (x / c) - 0.3516 * pow(x / c, 2) + 0.2843 * pow(x / c, 3) - 0.1015 * pow(x / c, 4));
        points.push_back({x + pos.x - c / 2, y + pos.y});
    }
    for (int i = n; i >= 0; i--)
    {
        double x = c * pow(double(i) / n, contraction);
        double y = (c * t / 0.2) * (0.2969 * sqrt(x / c) - 0.1260 * (x / c) - 0.3516 * pow(x / c, 2) + 0.2843 * pow(x / c, 3) - 0.1015 * pow(x / c, 4));
        points.push_back({x + pos.x - c / 2, -y + pos.y});
    }

    return rotatePolygon(points, pos, incidence);
}

vector<wxPoint> generateNacaProfile(wxPoint pos, double chord, double thickness, int nb_points, double incidence)
{
    vector<wxPoint> points;
    int contraction = 3;
    double c = chord;
    double t = thickness;
    int n = nb_points;

    for (int i = 0; i <= n; i++)
    {
        double x = c * pow(double(i) / n, contraction);
        double y = (c * t / 0.2) * (0.2969 * sqrt(x / c) - 0.1260 * (x / c) - 0.3516 * pow(x / c, 2) + 0.2843 * pow(x / c, 3) - 0.1015 * pow(x / c, 4));
        points.push_back(wxPoint(x + pos.x - c / 2, y + pos.y));
    }
    for (int i = n; i >= 0; i--)
    {
        double x = c * pow(double(i) / n, contraction);
        double y = (c * t / 0.2) * (0.2969 * sqrt(x / c) - 0.1260 * (x / c) - 0.3516 * pow(x / c, 2) + 0.2843 * pow(x / c, 3) - 0.1015 * pow(x / c, 4));
        points.push_back(wxPoint(x + pos.x - c / 2, -y + pos.y));
    }

    return rotatePolygon(points, pos, incidence);
}

vector<vector<RTCFDPoint>> generateRotor(RTCFDPoint center, double radius, double chord, double thickness, int nb_points, int Z)
{
    vector<vector<RTCFDPoint>> rotor;
    RTCFDPoint pos = {0, 0};

    generateNacaProfile(pos, chord, thickness, nb_points, -M_PI / 12);

    return rotor;
}

vector<vector<wxPoint>> generateRotor(wxPoint center, double radius, double chord, double thickness, int nb_points, int Z)
{
    vector<vector<wxPoint>> rotor;
    wxPoint pos = center;// {centre.x - radius, 0};

    generateNacaProfile(pos, chord, thickness, nb_points, -M_PI / 12);

    return rotor;
}

vector<vector<RTCFDPoint>> generateRotorPoints(RTCFDPoint pos, double length)
{
    return generateRotor(pos, 5.0 * length, length, 0.12, 10, 3);
}

vector<vector<wxPoint>> generateRotorPoints(wxPoint pos, double length)
{
    return generateRotor(pos, 5.0 * length, length, 0.12, 10, 3);
}

vector<RTCFDPoint> rotatePolygon(vector<RTCFDPoint> polygon, RTCFDPoint center, double theta)
{
    vector<RTCFDPoint> rotatedPolygon;
    for (int i = 0; i < polygon.size(); i++)
    {
        double x = center.x + (polygon[i].x - center.x) * cos(theta) - (polygon[i].y - center.y) * sin(theta);
        double y = center.y + (polygon[i].x - center.x) * sin(theta) + (polygon[i].y - center.y) * cos(theta);
        rotatedPolygon.push_back({x, y});
    }

    return rotatedPolygon;
}

vector<wxPoint> rotatePolygon(vector<wxPoint> polygon, wxPoint center, double theta)
{
    vector<wxPoint> rotatedPolygon;
    for (int i = 0; i < polygon.size(); i++)
    {
        double x = center.x + (polygon[i].x - center.x) * cos(theta) - (polygon[i].y - center.y) * sin(theta);
        double y = center.y + (polygon[i].x - center.x) * sin(theta) + (polygon[i].y - center.y) * cos(theta);
        rotatedPolygon.push_back(wxPoint(x, y));
    }

    return rotatedPolygon;
}

vector<vector<RTCFDPoint>> generateCircularRepeats(vector<RTCFDPoint> polygon, RTCFDPoint center, int n)
{
    vector<vector<RTCFDPoint>> circularRepeats;
    for (int i = 0; i < n; i++)
    {
        double theta = 2 * M_PI * i / n;
        vector<RTCFDPoint> rotatedPolygon;
        for (int j = 0; j < polygon.size(); j++)
        {
            double x = center.x + (polygon[j].x - center.x) * cos(theta) - (polygon[j].y - center.y) * sin(theta);
            double y = center.y + (polygon[j].x - center.x) * sin(theta) + (polygon[j].y - center.y) * cos(theta);
            rotatedPolygon.push_back({x, y});
        }
        circularRepeats.push_back(rotatedPolygon);
    }

    return circularRepeats;
}

