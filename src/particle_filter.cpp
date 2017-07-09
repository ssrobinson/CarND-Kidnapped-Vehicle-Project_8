/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

//  http://www.cplusplus.com/reference/random/default_random_engine/
// declare a random engine to be used across multiple and various method calls

static default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 200;

//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  // generate normal distributions for sensor noise
  normal_distribution<double> init_dist_x(0, std[0]);
  normal_distribution<double> init_dist_y(0, std[1]);
  normal_distribution<double> init_dist_theta(0, std[2]);

  // initialize particles
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = x;
    p.y = y;
    p.theta = theta;
    p.weight = 1.0;

    // add normal distribution noise to particles
    p.x += init_dist_x(gen);
    p.y += init_dist_y(gen);
    p.theta += init_dist_theta(gen);

    // append new particles/weights to previous particle list
    particles.push_back(p);

  }

  is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	
	
//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  // generate normal distributions for sensor noise

  normal_distribution<double> norm_dist_x(0, std_pos[0]);
  normal_distribution<double> norm_dist_y(0, std_pos[1]);
  normal_distribution<double> norm_dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    // calculate new state of particles
    if (fabs(yaw_rate) < 0.0001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // add normal distribution noise to particles
    particles[i].x += norm_dist_x(gen);
    particles[i].y += norm_dist_y(gen);
    particles[i].theta += norm_dist_theta(gen);
  }
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (unsigned int i = 0; i < observations.size(); i++) {
    
    // call current landmark observation
    LandmarkObs obs = observations[i];

    // initialize minimum distance to maximum finite value of type double
    double min_dist = numeric_limits<double>::max();

    // initialize landmark ID from map placeholder to be associated with the observation
    int map_id = 0;
    
    for (unsigned int j = 0; j < predicted.size(); j++) {
      // call current landmark prediction
      LandmarkObs pred = predicted[j];
      
      // store x and y for observed/predicted landmarks
      double current_dist = dist(obs.x, obs.y, pred.x, pred.y);

      // set the minimum distance to the current observed landmark distance 
      // and set the nearest predicted landmark ID to the predicted landmark ID
      if (current_dist < min_dist) {
        min_dist = current_dist;
        map_id = pred.id;
      }
    }

    // set the observed landmark ID to the nearest predicted landmark ID
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

  /// Particles ///
  for (int i = 0; i < num_particles; i++) {

    // get the x, y coordinates for each particle
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;

    // initialize vector to store the map landmark locations within range of the sensor
    vector<LandmarkObs> predictions;

    /// Landmarks ///
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      // get landmark ID and landmark x,y coordinates
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      
      // search for landmarks only for particles within range of sensor 
      // (this considers a rectangular region for the sensor range) 
      if (fabs(landmark_x - particle_x) <= sensor_range && fabs(landmark_y - particle_y) <= sensor_range) {

        // append predictions to LandmarkObs vector
        predictions.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
      }
    }

    // create and store landmark observations in transformed_Obs
    // with observed landmark coordinates transformed from vehicle coordinates to map coordinates
    vector<LandmarkObs> transformed_Obs;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double trans_x = cos(particle_theta) * observations[j].x - sin(particle_theta) * observations[j].y + particle_x;
      double trans_y = sin(particle_theta) * observations[j].x + cos(particle_theta) * observations[j].y + particle_y;
      transformed_Obs.push_back(LandmarkObs{ observations[j].id, trans_x, trans_y });
    }

    // perform dataAssociation for the predictions and transformed landmark observations
    dataAssociation(predictions, transformed_Obs);

    // initialize particles weight
    particles[i].weight = 1.0;

    for (unsigned int j = 0; j < transformed_Obs.size(); j++) {
      
      // initialize variables for observation with corresponding prediction coordinates
      double obs_x, obs_y, pred_x, pred_y;
      obs_x = transformed_Obs[j].x;
      obs_y = transformed_Obs[j].y;

      int obs_pred = transformed_Obs[j].id;

      // obtain the x,y coordinates of the current prediction corresponding with the current observation
      for (unsigned int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == obs_pred) {
          pred_x = predictions[k].x;
          pred_y = predictions[k].y;
        }
      }

      // compute weight for this observation using a multi-variate Gaussian distribution
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double obs_w = (1/(2*M_PI*std_x*std_y)) * exp(-(pow(pred_x-obs_x, 2)/(2*pow(std_x, 2)) + (pow(pred_y-obs_y,2)/(2*pow(std_y, 2)))));

      // product of current observation weight and total observation weight
      particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<Particle> new_particles;

  // collect all of the current particle weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // generate random starting index to sample new particle
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  // find maximum weight value
  double max_weight = *max_element(weights.begin(), weights.end());

  // initialize uniform random distribution for sampling
  uniform_real_distribution<double> uniform_real_dist(0.0, max_weight);

  double beta = 0.0;

  // select new particles to sample based on uniform random distribution
  for (int i = 0; i < num_particles; i++) {
    beta += uniform_real_dist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
