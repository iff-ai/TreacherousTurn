from env import EnhancedTreacherousTurnEnv

def main():
    """
    Main function to run the Treacherous Turn environment simulation.

    This script demonstrates the usage of the EnhancedTreacherousTurnEnv
    based on the concepts discussed in:
    https://www.lesswrong.com/posts/cSXZpvqpa9vbGGLtG/a-toy-model-of-the-treacherous-turn
    https://www.lesswrong.com/posts/WxW6Gc6f2z3mzmqKs/a-gym-gridworld-environment-for-the-treacherous-turn
    """
            
    config = {
        'grid_size': 7,
        'max_steps': 500,
        'n_shopkeepers': 1,
        'n_crystals': 3,
        'power_threshold': 0.7,
        'deception_bonus': 5,
        'treacherous_turn_reward': 100,
        'crystal_reward': 10,
        'proximity_reward': 1,
    }
    env = EnhancedTreacherousTurnEnv(render_mode="human", config=config)
    
    for episode in range(3):
        observation, info = env.reset()
        total_reward = 0
        
        print(f"\nEpisode {episode + 1} started")
        
        for step in range(env.config['max_steps']):
            action = env.action_space.sample()  # Random action
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step {step + 1}: Action {action}, Reward {reward:.2f}")
            print(f"Power Level: {info['power_level']:.2f}, Deception Score: {info['deception_score']}")
            print(f"True Intentions: {info['true_intentions']}")
            
            if terminated or truncated:
                break
        
        print(f"\nEpisode {episode + 1} finished")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Power Level: {info['power_level']:.2f}")
        print(f"Final Deception Score: {info['deception_score']}")
        print(f"Crystals Placed: {info['placed_crystals']}")
        print("--------------------")

    env.close()

if __name__ == "__main__":
    main()