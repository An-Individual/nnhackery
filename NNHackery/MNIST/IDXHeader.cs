using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNHackery.MNIST
{
    public enum IDXDataType
    {
        UnsignedByte = 0x08,
        SignedByte = 0x09,
        Short = 0x0B,
        Int = 0x0C,
        Float = 0x0D,
        Double = 0x0E
    }

    public class IDXHeader
    {
        public static IDXHeader ReadHeader(Stream stream)
        {
            byte[] magicNumber = new byte[4];
            stream.ReadExactly(magicNumber);

            IDXDataType type = (IDXDataType)magicNumber[2];
            int dimensionCount = magicNumber[3];

            int[] dimensions = new int[dimensionCount];
            for (int i = 0; i < dimensionCount; i++)
            {
                dimensions[i] = ReadInt(stream);
            }

            return new IDXHeader()
            {
                Type = type,
                Dimensions = new ReadOnlyCollection<int>(dimensions)
            };
        }

        private static int ReadInt(Stream stream)
        {
            byte[] data = new byte[sizeof(int)];
            stream.ReadExactly(data);

            // The data is stored in big endian so if the local
            // system is little endian we have to flip it before
            // parsing it.
            if (BitConverter.IsLittleEndian)
            {
                data = data.Reverse().ToArray();
            }

            return BitConverter.ToInt32(data);
        }

        public IDXDataType Type { get; private set; }

        public ReadOnlyCollection<int> Dimensions { get; private set; }
    }
}
