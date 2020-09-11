/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    
    /**
    * TODO: Set the number of particles. Initialize all particles to
    *   first position (based on estimates of x, y, theta and their uncertainties
    *   from GPS) and all weights to 1.
    * TODO: Add random Gaussian noise to each particle.
    * NOTE: Consult particle_filter.h for more information about this method
    *   (and others in this file).
    */
    
    std::default_random_engine gen;
    double std_x, std_y, std_theta;  // StdDev for x, y, and theta

    // Set standard deviations for x, y, and theta
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    // This line creates a normal (Gaussian) distribution
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> angle_theta(theta, std_theta);

    for (int i = 0; i < 3; ++i) {
      
        Particle p;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = angle_theta(gen);
        p.weight = 1;
        
        particles.push_back(p);
    }

    num_particles = particles.size();  // TODO: Set the number of particles

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
    
    /*
    * TODO: Add measurements to each particle and add random Gaussian noise.
    * NOTE: When adding noise you may find std::normal_distribution
    *   and std::default_random_engine useful.
    *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    *  http://www.cplusplus.com/reference/random/default_random_engine/
    */
    
    std::default_random_engine gen;
    double std_x, std_y, std_theta;
    double theta_dot;
    
    // Set standard deviations for x, y, and theta
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];
    
    // calculate theta_dot
    theta_dot = yaw_rate/delta_t;

    for (int i = 0; i< num_particles; i++){
        
        double theta0 = particles[i].theta;
        particles[i].x += (velocity/theta_dot) * (sin(theta0+theta_dot*delta_t) - sin(theta0));
        particles[i].y += (velocity/theta_dot) * (cos(theta0) - cos(theta0+theta_dot*delta_t));
        particles[i].theta += theta_dot*delta_t;
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /*
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
    /*
    * TODO: Update the weights of each particle using a mult-variate Gaussian
    *   distribution. You can read more about this distribution here:
    *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     
    * NOTE: The observations are given in the VEHICLE'S coordinate system.
    *   Your particles are located according to the MAP'S coordinate system.
    *   You will need to transform between the two systems. Keep in mind that
    *   this transformation requires both rotation AND translation (but no scaling).
    *   The following is a good resource for the theory:
    *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    *   and the following is a good resource for the actual equation to implement
    *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
    */
    double weight = 1;
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    
    for (int i = 0; i< num_particles; i++){
        
        //get all data variables ready
        double x_map, y_map, x_obs, y_obs, mu_x, mu_y;
        double x_part = particles[i].x;
        double y_part = particles[i].y;
        double theta = particles[i].theta;
        
        // loop observations
        for (int j=0; j< observations.size(); j++){
            x_obs = observations[j].x;
            y_obs = observations[j].y;
            
            // transform obs to the particle's coordinate system
            x_map = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
            y_map = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);
            
            //loop map to find the closest landmark's coordinates
            double dist_min = sensor_range;
            for (int k=0; k<map_landmarks.landmark_list.size();k++){
                Map::single_landmark_s lm = map_landmarks.landmark_list[k];
                double dist_t = dist(x_map, y_map, lm.x_f, lm.y_f);
                if (dist_t < dist_min){
                    mu_x = lm.x_f;
                    mu_y = lm.y_f;
                }
            }
            
            // calculate weight using mult-variate Gaussian
            double gauss_norm, exponent;
            gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
            exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
                         + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
            weight = gauss_norm * exp(-exponent);
            
            // multiply the weight
            particles[i].weight *= weight;
        }
    }
    
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    std::vector<Particle> particles_resample;
    std::vector<float> W_vect;
    
    for(int n=0; n< num_particles; ++n) {
        W_vect.push_back(particles[n].weight);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(W_vect.begin(),W_vect.end());
    
    for(int n=0; n< num_particles; ++n) {
        //++m[d(gen)];
        int idx = d(gen);
        Particle p = particles[idx];
        particles_resample.push_back(p);
    }
    
    particles = particles_resample;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
