functions {

  vector saa_total(vector[] saa_norm_vec, vector[] saa_decay_vec, matrix[] t_t0, int num_data_points,
                   int num_saa_exits){
    vector[num_data_points] total_saa_counts = rep_vector(0.0, num_data_points);
    for (i in 1:num_saa_exits){
      total_saa_counts += saa_norm_vec[i]./saa_decay_vec[i].*
        (exp(-t_t0[i,:,1].*saa_decay_vec[i]) - exp(-t_t0[i,:, 2].*saa_decay_vec[i, :]));
    }
    return total_saa_counts;

  }

  real partial_sum(int[] counts, int start, int stop,
                   vector[] base_counts_array,
                   real[] norm_fixed,
                   vector[] norm_cont_vec, vector[] base_counts_array_cont,
                   matrix[] t_t0, vector[] saa_decay_vec, vector[] saa_norm_vec){

    return poisson_propto_lpmf(counts |
                               // SAA Block
                               saa_norm_vec[1, start:stop]./saa_decay_vec[1, start:stop].*
                               (exp(-t_t0[1,start:stop,1].*saa_decay_vec[1, start:stop])-
                                exp(-t_t0[1,start:stop,2].*saa_decay_vec[1, start:stop]))+

                               saa_norm_vec[2, start:stop]./saa_decay_vec[2, start:stop].*
                               (exp(-t_t0[2,start:stop,1].*saa_decay_vec[2, start:stop])-
                                exp(-t_t0[2,start:stop,2].*saa_decay_vec[2, start:stop]))+

                               saa_norm_vec[3, start:stop]./saa_decay_vec[3, start:stop].*
                               (exp(-t_t0[3,start:stop,1].*saa_decay_vec[3, start:stop])-
                                exp(-t_t0[3,start:stop,2].*saa_decay_vec[3, start:stop]))+

                               saa_norm_vec[4, start:stop]./saa_decay_vec[4, start:stop].*
                               (exp(-t_t0[4,start:stop,1].*saa_decay_vec[4, start:stop])-
                                exp(-t_t0[4,start:stop,2].*saa_decay_vec[4, start:stop]))+

                               saa_norm_vec[5, start:stop]./saa_decay_vec[5, start:stop].*
                               (exp(-t_t0[5,start:stop,1].*saa_decay_vec[5, start:stop])-
                                exp(-t_t0[5,start:stop,2].*saa_decay_vec[5, start:stop]))+

                               saa_norm_vec[6, start:stop]./saa_decay_vec[6, start:stop].*
                               (exp(-t_t0[6,start:stop,1].*saa_decay_vec[6, start:stop])-
                                exp(-t_t0[6,start:stop,2].*saa_decay_vec[6, start:stop]))+

                               saa_norm_vec[7, start:stop]./saa_decay_vec[7, start:stop].*
                               (exp(-t_t0[7,start:stop,1].*saa_decay_vec[7, start:stop])-
                                exp(-t_t0[7,start:stop,2].*saa_decay_vec[7, start:stop]))+

                               saa_norm_vec[8, start:stop]./saa_decay_vec[8, start:stop].*
                               (exp(-t_t0[8,start:stop,1].*saa_decay_vec[8, start:stop])-
                                exp(-t_t0[8,start:stop,2].*saa_decay_vec[8, start:stop]))+

                               saa_norm_vec[9, start:stop]./saa_decay_vec[9, start:stop].*
                               (exp(-t_t0[9,start:stop,1].*saa_decay_vec[9, start:stop])-
                                exp(-t_t0[9,start:stop,2].*saa_decay_vec[9, start:stop]))+

                               saa_norm_vec[10, start:stop]./saa_decay_vec[10, start:stop].*
                               (exp(-t_t0[10,start:stop,1].*saa_decay_vec[10, start:stop])-
                                exp(-t_t0[10,start:stop,2].*saa_decay_vec[10, start:stop]))+

                               norm_fixed[1]*base_counts_array[1,start:stop]+
                               norm_fixed[2]*base_counts_array[2,start:stop]+
                               norm_fixed[3]*base_counts_array[3,start:stop]+
                               norm_fixed[4]*base_counts_array[4,start:stop]+
                               norm_fixed[5]*base_counts_array[5,start:stop]+
                               norm_fixed[6]*base_counts_array[6,start:stop]+
                               //norm_fixed[7]*base_counts_array[7,start:stop]+
                               //norm_fixed[8]*base_counts_array[8,start:stop]+
                               //norm_fixed[9]*base_counts_array[9,start:stop]+
                               //norm_fixed[10]*base_counts_array[10,start:stop]+
                               norm_cont_vec[1][start:stop].*base_counts_array_cont[1][start:stop]+
                               norm_cont_vec[2][start:stop].*base_counts_array_cont[2][start:stop]);
  }

}


data {

  // num time bins, dets and echans
  int<lower=1> num_time_bins;
  int<lower=1> num_dets;
  int<lower=1> num_echans;

  // Number of free pointsources and fixed sources
  int<lower=0> num_fixed_comp;

  // Energy input bins for response calc.
  int<lower=1> rsp_num_Ein;
  vector[rsp_num_Ein] Ebins_in[2];

  int grainsize;

  // SAA exits - Prototype
  int num_saa_exits;
  vector[num_saa_exits] saa_start_times;

  // Time bins
  matrix[num_time_bins, 2] time_bins;

  // Observed counts - vector structure [timebin1&det1&echan1,timebin1&det1&echan2,...,timebin1&det2&echan1,timebin1&det2&echan2,...,timebin2&det1&echan1,timebin2&det1&echan2,...,timebinN&detM&echanK]
  int counts[num_time_bins*num_dets*num_echans];

  // For global sources give the precalculated counts arrays (Physical photon sources with fixed spectrum) - only a normalization is fitted
  // One vector per global source.
  vector[num_time_bins*num_dets*num_echans] base_counts_array[num_fixed_comp];

  // Cont sources
  int num_cont_comp;
  vector[num_time_bins*num_dets*num_echans] base_counts_array_cont[num_cont_comp];
}

transformed data {

  int num_data_points = num_time_bins*num_dets*num_echans;
  matrix[num_data_points,2] t_t0[num_saa_exits];
  int time_bin_indices[num_time_bins];
  int bin_det_echan[num_dets, num_echans,num_time_bins];
  int use_saa;

  // Check if there is at least one saa exit
  if (num_saa_exits>0){
    use_saa=1;
  }
  else{
    use_saa=0;
  }

  for (j in 1:num_saa_exits){
    for (i in 1:num_time_bins){
      if (time_bins[i,1]>saa_start_times[j]){
          t_t0[j,(i-1)*num_dets*num_echans+1:i*num_dets*num_echans] = rep_matrix(time_bins[i]-saa_start_times[j], num_dets*num_echans);
        }
      else {
        t_t0[j,(i-1)*num_dets*num_echans+1:i*num_dets*num_echans] = rep_matrix(0.0, num_dets*num_echans, 2);
      }
    }
  }

  // Calc length of all time bins
  //time_bin_length = (time_bins[:,2]-time_bins[:,1]);

  // Get an array with [1,2,...,num_time_bins]
  for (i in 1:num_time_bins){
    time_bin_indices[i] = i;
  }

  for (i in 1:num_dets){
        for (j in 1:num_echans){
          for (t in 1:num_time_bins){
            bin_det_echan[i,j,t] = (t-1)*num_dets*num_echans+(j-1)+(i-1)*num_echans+1;
          }
        }
      }
}

parameters {
  // Norm of fixed components
  real<lower=0> norm_fixed[num_fixed_comp];

  // SAA parameter
  real<lower=0> norm_saa[use_saa ? num_saa_exits:0,num_dets, num_echans];
  real<lower=0.01,upper=10> decay_saa[use_saa ? num_saa_exits:0,num_dets, num_echans];

  real<lower=0> norm_cont[num_cont_comp, num_dets, num_echans];
}

transformed parameters {
  vector[num_data_points] norm_cont_vec[num_cont_comp];

  vector[num_data_points] saa_decay_vec[num_saa_exits];
  vector[num_data_points] saa_norm_vec[num_saa_exits];

  for (l in 1:num_cont_comp){
    for (i in 1:num_dets){
      for (j in 1:num_echans){
        for (k in 1:num_time_bins){
          norm_cont_vec[l][(k-1)*(num_dets*num_echans)+(i-1)*num_echans+j] = norm_cont[l,i,j];
        }
      }
    }
  }

  for (l in 1:num_saa_exits){
    for (i in 1:num_dets){
      for (j in 1:num_echans){
        for (k in 1:num_time_bins){
          saa_decay_vec[l][(k-1)*(num_dets*num_echans)+(i-1)*num_echans+j] = 0.0001*decay_saa[l,i,j];
          saa_norm_vec[l][(k-1)*(num_dets*num_echans)+(i-1)*num_echans+j] = norm_saa[l,i,j];
        }
      }
    }
  }
}

model {

  norm_fixed ~ lognormal(0,1);

  if (num_cont_comp>0){
    for (d in 1:num_dets){
      for (g in 1:num_echans){
        norm_cont[:,d,g] ~ lognormal(0,1);
      }
    }
  }

  if (use_saa==1){
    for (d in 1:num_dets){
      for (e in 1:num_echans){
         norm_saa[:,d,e] ~ lognormal(0,1);
         decay_saa[:,d,e] ~ lognormal(0,1);
      }
    }
  }

  // Call reduce_sum
  target += reduce_sum(partial_sum, counts, grainsize,
                       base_counts_array,  norm_fixed,
                       norm_cont_vec, base_counts_array_cont, t_t0, saa_decay_vec, saa_norm_vec);
}

generated quantities {

  // Calculate ppc as well as the contribution from earth, cgb, fixed sources,...
  int ppc[num_data_points];
  vector[num_data_points] tot;
  vector[num_data_points] f_fixed[num_fixed_comp];
  vector[num_data_points] f_saa;
  vector[num_data_points] f_cont[num_cont_comp];

  tot = rep_vector(0.0,num_data_points);

  for (i in 1:num_fixed_comp){
    f_fixed[i] = norm_fixed[i]*base_counts_array[i];
    tot += f_fixed[i];
  }

  if (use_saa==1){
    f_saa = saa_total(saa_norm_vec, saa_decay_vec, t_t0, num_data_points, num_saa_exits);
    tot += f_saa;
  }

  for (i in 1:num_cont_comp){
    for (j in 1:num_dets){
      for (k in 1:num_echans){
        f_cont[i][bin_det_echan[j,k]] = norm_cont[i,j,k]*base_counts_array_cont[i][bin_det_echan[j,k]];
      }
    }
    tot += f_cont[i];
  }

  ppc = poisson_rng(tot);

}
