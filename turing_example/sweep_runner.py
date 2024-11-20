# Import the W&B Python Library and log into W&B
import wandb
import click
import random
import math

# 1: Define objective/training function
def objective(config,i):
    score = math.sqrt(config['coolparameter1']**3/(i+1)) + config['coolparameter2'] + random.uniform(0,0.5)
    return score

@click.command()
@click.option("--coolparameter1", default=1.0)
@click.option("--coolparameter2", default=1.0)
@click.option("--lr", default=0.01)
@click.option("--modeltype", default='mymodel-type')
@click.option("--epochs", default=100)
def main(**kw):
    wandb.init(project="cs533-turing-example", config=kw)
    for i in range(kw['epochs']): 
        score = objective(kw,i)
        wandb.log({"score": score})

if __name__=="__main__":
    main()    
