/*
 * This file is part of the Substack Analysis (SUSAN) framework.
 * Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
 * Max Planck Institute of Biophysics
 * Department of Structural Biology - Kudryashev Group.
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

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
    static Vec3 *cuboid(uint32&counter, const float x_range, const float y_range, const float z_range, const float step) {

        if( (x_range+y_range+z_range) == 0 ) {
            counter = 1;
            Vec3 *points = new Vec3[counter];
            points[0].x = 0;
            points[0].y = 0;
            points[0].z = 0;
            return points;
        }

        float x,y,z;
        counter=0;

        float x_lim = step*floor(x_range/step);
        float y_lim = step*floor(y_range/step);
        float z_lim = step*floor(z_range/step);

        for(z=-z_lim; z<=z_lim; z=z+step) {
            for(y=-y_lim; y<=y_lim; y=y+step) {
                for(x=-x_lim; x<=x_lim; x=x+step) {
                    counter++;
                }
            }
        }

        Vec3 *points = new Vec3[counter];
        counter=0;
        for(z=-z_lim; z<=z_lim; z=z+step) {
            for(y=-y_lim; y<=y_lim; y=y+step) {
                for(x=-x_lim; x<=x_lim; x=x+step) {
                    points[counter].x = x;
                    points[counter].y = y;
                    points[counter].z = z;
                    counter++;
                }
            }
        }

        return points;
    }

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

        if( (x_range+y_range+z_range) == 0 ) {
            counter = 1;
            Vec3 *points = new Vec3[counter];
            points[0].x = 0;
            points[0].y = 0;
            points[0].z = 0;
            return points;
        }

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

    static Vec3 *circle(uint32&counter, const float x_range, const float y_range, const float step) {

        if( (x_range+y_range) == 0 ) {
            counter = 1;
            Vec3 *points = new Vec3[counter];
            points[0].x = 0;
            points[0].y = 0;
            points[0].z = 0;
            return points;
        }

        float x,y,X,Y;
        counter=0;

        float x_lim = floor(x_range);
        float y_lim = floor(y_range);

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

        Vec3 *points = new Vec3[counter];
        counter=0;
        for(y=-y_lim; y<=y_lim; y=y+step) {
            Y = y/y_lim;
            for(x=-x_lim; x<=x_lim; x=x+step) {
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



