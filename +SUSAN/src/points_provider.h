#ifndef POINTS_PROVIDER_H
#define POINTS_PROVIDER_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "datatypes.h"

using namespace Eigen;

class PointsProvider {

public:
    static Vec3 *ellipsoid(uint32&counter, const float x_range, const float y_range, const float z_range, const float step) {

                if( (x_range+y_range+z_range) == 0 ) {
                    counter = 1;
                    Vec3 *points = new Vec3[counter];
                    points[0].x = 0;
                    points[0].y = 0;
                    points[0].z = 0;
                    return points;
                }

		float x,y,z,X,Y,Z;
		counter=0;

		float x_lim = step*floor(x_range/step);
		float y_lim = step*floor(y_range/step);
		float z_lim = step*floor(z_range/step);

		for(z=-z_lim; z<=z_lim; z=z+step) {
			Z = z/z_lim;
			for(y=-y_lim; y<=y_lim; y=y+step) {
				Y = y/y_lim;
				for(x=-x_lim; x<=x_lim; x=x+step) {
					X = x/x_range;
					float R = sqrt( X*X + Y*Y + Z*Z );
					if( R <= 1.0f ) {
						counter++;
					}
				}
			}
		}

		Vec3 *points = new Vec3[counter];
		counter=0;
		for(z=-z_lim; z<=z_lim; z=z+step) {
			Z = z/z_lim;
			for(y=-y_lim; y<=y_lim; y=y+step) {
				Y = y/y_lim;
				for(x=-x_lim; x<=x_lim; x=x+step) {
					X = x/x_lim;
					float R = sqrt( X*X + Y*Y + Z*Z );
					if( R <= 1.0f ) {
						points[counter].x = x;
						points[counter].y = y;
						points[counter].z = z;
						counter++;
					}
				}
			}
		}

		return points;
	}

    static Vec3 *sphere(uint32&counter, const float radius, const float step) {
		return ellipsoid(counter,radius,radius,radius,step);
	}

    static Vec3 *cylinder(uint32&counter, const float x_range, const float y_range, const float z_range, const float step) {

		float x,y,z,X,Y;
		counter=0;

		float x_lim = step*floor(x_range/step);
		float y_lim = step*floor(y_range/step);
		float z_lim = step*floor(z_range/step);

		for(z=-z_lim; z<=z_lim; z=z+step) {
			for(y=-y_lim; y<=y_lim; y=y+step) {
				Y = y/y_lim;
				for(x=-x_lim; x<=x_lim; x=x+step) {
					X = x/x_lim;
					float R = sqrt( X*X + Y*Y );
					if( R <= 1.0f ) {
						counter++;
					}
				}
			}
		}

		Vec3 *points = new Vec3[counter];
		counter=0;
		for(z=-z_lim; z<=z_lim; z=z+step) {
			for(y=-y_lim; y<=y_lim; y=y+step) {
				Y = y/y_lim;
				for(x=-x_lim; x<=x_lim; x=x+step) {
					X = x/x_lim;
					float R = sqrt( X*X + Y*Y );
					if( R <= 1.0f ) {
						points[counter].x = x;
						points[counter].y = y;
						points[counter].z = z;
						counter++;
					}
				}
			}
		}

		return points;
	}

    static Vec3 *circle(uint32&counter, const float x_range, const float y_range) {

		float x,y,X,Y;
		counter=0;

		float x_lim = floor(x_range);
		float y_lim = floor(y_range);

		for(y=-y_lim; y<=y_lim; y=y+1.0f) {
			Y = y/y_lim;
			for(x=-x_lim; x<=x_lim; x=x+1.0f) {
				X = x/x_lim;
				float R = sqrt( X*X + Y*Y );
				if( R <= 1.0f ) {
					counter++;
				}
			}
		}

		Vec3 *points = new Vec3[counter];
		counter=0;
		for(y=-y_lim; y<=y_lim; y=y+1.0f) {
			Y = y/y_lim;
			for(x=-x_lim; x<=x_lim; x=x+1.0f) {
				X = x/x_lim;
				float R = sqrt( X*X + Y*Y );
				if( R <= 1.0f ) {
					points[counter].x = x;
					points[counter].y = y;
					points[counter].z = 0;
					counter++;
				}
			}
		}

		return points;
	}

};

#endif /// POINTS_PROVIDER_H



