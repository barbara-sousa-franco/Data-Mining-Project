Working with GitHub
STEP 1: Before starting, confirm your at your branch

git branch
If you are at your branch, go straight to step 2. If not, change to your branch:

git checkout nome-do-teu-branch
STEP 2:

git pull origin nome-branch-comum
Now you're ready to work
STEP 3: When you finish

Save your work (Control+S)
git add .
git commit -m "message"
git push origin your-branch
STEP 4: Atualize main

git checkout main
git pull origin main
git merge your-branch
git push origin main
