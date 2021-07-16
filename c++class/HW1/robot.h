#ifndef GROUP_ROBOT_INCLUDED
#define GROUP_ROBOT_INCLUDED

class Robot {
public:
    Robot();
    void predictNextPos();
    void turnRight();
    void setPosition(const int x, const int y);
    void setNextPosition(const int x, const int y);
    void setTotalSteps(const long long totalSteps);
    const int getRobotDir() const;
    const int getX() const;
    const int getY() const;
    const int getNextX() const;
    const int getNextY() const;
    const long long getTotalSteps() const;
private:    
    static constexpr int NORTH = 0;
    static constexpr int EAST = 1;
    static constexpr int SOUTH = 2;
    static constexpr int WEST = 3;
    int x;
    int y;
    int nextX; 
    int nextY; 
    int robotDir;
    long long totalSteps;
};

#endif