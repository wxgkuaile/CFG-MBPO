import numpy as np
import torch

class PredictEnv:
    def __init__(self, args, model, env_name, model_type):
        self.model = model
        self.env_name = env_name
        self.model_type = model_type
        self.args = args
        self.device = args.device

    def get_model(self):
        return self.model

    def _termination_fn(self, env_name, obs, act, next_obs):
        # TODO
        if env_name == "Hopper-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Walker2d-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Ant-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            x = next_obs[:, 0]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * (x >= 0.2) \
                       * (x <= 1.0)

            done = ~not_done
            done = done[:, None]
            return done

        elif env_name == "AntTruncatedObsEnv-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            x = next_obs[:, 0]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * (x >= 0.2) \
                       * (x <= 1.0)

            done = ~not_done
            done = done[:, None]
            return done

        elif env_name == "HalfCheetah-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "Humanoid-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            z = next_obs[:, 0]
            done = (z < 1.0) + (z > 2.0)

            done = done[:, None]
            return done
        elif env_name == "HumanoidTruncatedObsEnv-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            z = next_obs[:, 0]
            done = (z < 1.0) + (z > 2.0)

            done = done[:, None]
            return done
        elif env_name == "Swimmer-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "SwimmerTruncatedEnv-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "InvertedPendulum-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            notdone = np.isfinite(next_obs).all(axis=-1) \
                      * (np.abs(next_obs[:, 1]) <= .2)
            done = ~notdone

            done = done[:, None]

            return done

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        if self.model_type == 'pytorch':
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        if self.model_type == 'pytorch':
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        else:
            model_idxes = self.model.random_inds(batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        batch_size = model_means.shape[0]

        return_means = np.concatenate((model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info

    def sample(self, input_oa, deterministic=False):

        input_oa = input_oa.cpu().numpy()

        inputs = input_oa
        if self.model_type == 'pytorch':
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)

        ensemble_model_means[:, :, 1:] += inputs[:, :ensemble_model_means.shape[-1]-1]
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        if self.model_type == 'pytorch':
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        else:
            model_idxes = self.model.random_inds(batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        # rewards, next_obs = samples[:, :1], samples[:, 1:]
        # terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        # A = np.concatenate((rewards, next_obs), axis=-1)
        A = samples
        return A, log_prob, None

    def ParaMPCact(self, init_s, init_a, t, H, agent, rewarder):
        B = init_s.shape[0]  # 100000
        PlanningHorizon = int(min(int(self.args.MPCHorizon), H-t))

        init_s = torch.from_numpy(init_s).unsqueeze(0).repeat(self.args.n_trajs, 1, 1).view(self.args.n_trajs*B, -1)  # (n_trajs, B, _dim) 并行rollout
        init_a = torch.from_numpy(init_a).unsqueeze(0).repeat(self.args.n_trajs, 1, 1).view(self.args.n_trajs*B, -1)
        reward_list = torch.zeros(self.args.n_trajs, PlanningHorizon)
        for step in range(PlanningHorizon):
            if step == 0:
                s, a = init_s.numpy(), init_a.numpy()
            with torch.no_grad():
                if self.args.deter_model:
                    pred_next_s, pred_r, terminals = self.step_for_MPC(s, a)
                else:
                    pred_next_s, pred_r, _, _, terminals = self.step_for_MPC(s, a)  
                #pred_next_s, pred_r, _, _, terminals = self.step_for_MPC(s, a)
                
                if self.args.flow_option == 0:
                    input = torch.Tensor(np.concatenate((pred_next_s, pred_r), axis=-1)).to(self.device)  # (rollB*n_trajs, _dim)
                elif self.args.flow_option == 1:
                    pred_action = agent.select_action(s, eval=False)
                    input = torch.Tensor(np.concatenate((pred_next_s, pred_action, pred_r), axis=-1)).to(self.device)  # (rollB*n_trajs, _dim)

                # print("input: ", input.shape)  # input:  torch.Size([600000, 26])

                flow_reward = self.flow_get_reward(input, rewarder)  # compute flow reward
                gamma = self.args.model_gamma
                reward_list[:, step] = gamma**step * flow_reward.view(self.args.n_trajs, B, 1).mean(1).squeeze()
            nonterm_mask = ~terminals.squeeze(-1)
            if step == 0:
                j_first_preds = torch.from_numpy(pred_next_s).view(self.args.n_trajs, B, -1).numpy()
                j_first_predr = torch.from_numpy(pred_r).view(self.args.n_trajs, B, -1).numpy()
                j_first_ter = torch.from_numpy(terminals).view(self.args.n_trajs, B, -1).numpy()
            if nonterm_mask.sum() == 0:
                break
            s = pred_next_s
            a = agent.select_action(s, eval=True) if self.args.StoPoMPC else agent.select_action(s, eval=False)
        print("reward_list: ", reward_list)
        accumu_reward = reward_list.sum(-1)
        best_traj_num = accumu_reward.argmax()
        print("accumu_reward: ", accumu_reward)
        print("best_traj_num: ",best_traj_num)
        optimal_preds = j_first_preds[best_traj_num]
        optimal_predr = j_first_predr[best_traj_num]
        optimal_ter = j_first_ter[best_traj_num]
        return optimal_preds, optimal_predr, optimal_ter

    def flow_get_reward(self, input, rewarder):
        r = rewarder.get_reward(input, not_rl=True)
        return r

    def step_for_MPC(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
        inputs = np.concatenate((obs, act), axis=-1)

        #if self.args.deter_model:
        #    ensemble_model_means = self.model.deter_predict(inputs)
        #else:
        #    ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
            
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:, :, 1:] += obs
        if not self.args.deter_model:
            ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if self.args.deter_model:
            noise = (torch.randn_like(torch.from_numpy(ensemble_model_means)) * self.args.ClipDMoNoise).clamp(-self.args.ClipDMoNoise, self.args.ClipDMoNoise)
            ensemble_samples = ensemble_model_means + noise.numpy()
        else:
            ensemble_samples = ensemble_model_means # + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        batch_idxes = np.arange(0, batch_size)
        samples = ensemble_samples[model_idxes, batch_idxes]
        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        if self.args.deter_model:
            return next_obs, rewards, terminals
        else:
            en_mu_mean, en_sig_mean = ensemble_model_means.mean(0)[0], ensemble_model_stds.mean(0)[0] # (_dim,)
            return next_obs, rewards, en_mu_mean, en_sig_mean, terminals