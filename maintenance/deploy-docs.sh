#!/bin/bash
set -e
VERSION=v`python3 -c "import beat; print(beat.__version__);"`

if [ ! -f maintenance/deploy-docs.sh ] ; then
    echo "must be run from beat's toplevel directory"
    exit 1
fi

cd docs
rm -rf build/$VERSION
make clean; make html $1
cp -r build/html build/$VERSION

read -r -p "Are your sure to update live docs at http://pyrocko.org/beat/docs/$VERSION [y/N]?" resp
case $resp in
    [yY][eE][sS]|[yY] )
        rsync -av build/$VERSION pyrocko@hive.pyrocko.org:/var/www/pyrocko.org/beat/docs/;
        ;;
    * ) ;;
esac

read -r -p "Do you want to link 'current' to the just uploaded version $VERSION [y/N]?" resp
case $resp in
    [yY][eE][sS]|[yY] )
        echo "Linking /beat/docs/$VERSION to /beat/docs/current";
        ssh pyrocko@hive.pyrocko.org "rm -rf /var/www/pyrocko.org/beat/docs/current; ln -s /var/www/pyrocko.org/beat/docs/$VERSION /var/www/pyrocko.org/beat/docs/current";
        ;;
    * ) ;;
esac

cd ..
