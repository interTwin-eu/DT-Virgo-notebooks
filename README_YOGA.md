# Instructions to run on Yoga cluster

## Prepare your SSH setup
Create an SSH key to use with GitLab. Start with:

```
ssh-keygen -t ed25519 -C "<your email>" -f ~/.ssh/id_gitlab
```

Add the key to the SSH agent:

```
eval $(ssh-agent -s)
ssh-add ~/.ssh/id_gitlab
```

Lastly, configure your SSH agent to use the new key with gitlab.com.
Add the following lines to `~/.ssh/config`:

```
Host gitlab.com
  PreferredAuthentications publickey
  IdentityFile ~/.ssh/id_gitlab
```

## Clone the repo and setup the machine
Clone the remote repo:

```
cd <wherever you want>
git clone git@gitlab.com:edoardograsso98/intertwin_gwaves.git
```

Move to `./intertwin_gwaves/new_architectures/build_datasets` and clone additional data files there.
The source file is located here: https://unipiit-my.sharepoint.com/personal/a012554_unipi_it/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fa012554%5Funipi%5Fit%2FDocuments%2FSharedDocsOneDriveRazzano%2FInterTwin%2Ddati%2Daux&ga=1

Unfortunately the sharepoint directory is **NOT** accessible to external users.
