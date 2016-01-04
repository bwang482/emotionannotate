# -*- mode: ruby -*-
# vi: set ft=ruby :

# Vagrantfile API/syntax version. Don't touch unless you know what you're doing!
VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
  config.vm.box = "boxesio/wheezy64-ansible"

  config.vm.network "forwarded_port", guest: 9000, host: 9000

  config.ssh.forward_agent = true

config.vm.provision "ansible" do |ansible|
  ansible.playbook = "provision/vagrant.yml"
end

config.vm.provider "virtualbox" do |v|
  v.memory = 3096
  v.cpus = 4
end

end
