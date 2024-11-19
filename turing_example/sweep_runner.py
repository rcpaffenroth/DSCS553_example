# Import the W&B Python Library and log into W&B
import wandb
import click


# 1: Define objective/training function
def objective(config):
    score = config.x**3 + config.y
    return score

@click.command()
@click.option("--x", default=1.0)
@click.option("--y", default=1.0)
def main(x, y):
    wandb.init(project="cs533-turing-example")
    wandb.config['x'] = x
    wandb.config['y'] = y
    
    score = objective(wandb.config)
    wandb.log({"score": score})

if __name__=="__main__":
    main()    
